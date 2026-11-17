"""Physical 3R planar arm: shoulder/elbow/wrist, analytic IK and joint PD servos.

This is a configurable educational robot, not a manufacturer-calibrated arm.
Only reset initializes qpos. Every subsequent motion comes from joint actuators.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .upright import UprightEnv, wrap


class Arm3REnv(UprightEnv):
    def xml(self):
        root = ET.fromstring(super().xml())
        root.set('model', 'arm3r_upright')
        ET.SubElement(root.find('visual'), 'global', offwidth=str(self.c['render_width']), offheight=str(self.c['render_height']))
        world = root.find('worldbody')
        hand = world.find("body[@name='gripper']")
        c = self.c
        l1, l2, l3 = c['arm_lengths']
        hand.set('pos', f'0 0 {l3}')
        hand.set('quat', '0 0 1 0')
        ET.SubElement(hand, 'site', name='tcp', pos='0 0 0', size='0.003', rgba='0 0 0 0')
        world.remove(hand)
        bx, bz = c['arm_base']
        base = ET.SubElement(world, 'body', name='arm_base', pos=f'{bx} 0 0')
        ET.SubElement(base, 'geom', name='base_geom', type='cylinder', pos=f'0 0 {bz/2}', size=f'0.045 {bz/2}', rgba='0.18 0.21 0.25 1')
        parent = base
        for i, name in enumerate(['shoulder','elbow','arm_wrist']):
            offset = bz if i==0 else c['arm_lengths'][i-1]
            body = ET.SubElement(parent, 'body', name=f'arm_link{i+1}', pos=f'0 0 {offset}')
            ET.SubElement(body, 'joint', name=name, type='hinge', axis='0 1 0', limited='true',
                          range=' '.join(map(str,c['arm_joint_ranges'][i])), damping='0.05', armature=str(c['arm_armature'][i]))
            length = c['arm_lengths'][i] if i<2 else l3-0.032
            ET.SubElement(body, 'geom', name=f'arm_geom{i+1}', type='capsule', fromto=f'0 0 0 0 0 {length}',
                          size=str(c['arm_radii'][i]), mass=str(c['arm_masses'][i]), rgba='0.83 0.86 0.89 1')
            ET.SubElement(body, 'geom', name=f'joint_cover{i+1}', type='cylinder', size=f'{c["arm_radii"][i]*1.4} 0.025',
                          quat='0.70710678 0.70710678 0 0', mass='0.025', rgba='0.15 0.35 0.65 1')
            parent = body
        parent.append(hand)
        actuators = root.find('actuator')
        for i,name in reversed(list(enumerate(['shoulder','elbow','arm_wrist']))):
            actuator = ET.Element('position', name=f'{name}_servo', joint=name, kp=str(c['arm_kp'][i]), kv=str(c['arm_kv'][i]),
                                 forcerange=f'{-c["arm_torque"][i]} {c["arm_torque"][i]}',
                                 ctrllimited='true', ctrlrange=' '.join(map(str,c['arm_joint_ranges'][i])))
            actuators.insert(0,actuator)
        # Adjacent mounting parts overlap at the shoulder bearing by construction.
        contacts = ET.SubElement(root, 'contact')
        ET.SubElement(contacts, 'exclude', body1='arm_base', body2='arm_link1')
        # All non-adjacent robot links and scene geometry retain collision checks.
        world.find("camera[@name='front']").set('pos','0.12 -1.05 0.51')
        world.find("camera[@name='front']").set('xyaxes','1 0 0 0 0.22 1')
        return ET.tostring(root, encoding='unicode')

    def inverse_kinematics(self, position, theta):
        c = self.c
        l1,l2,l3 = c['arm_lengths']
        # Arm wrist pivot is behind the gripper by l3 along gripper +z.
        wrist_pos = np.asarray(position)+l3*np.array([np.sin(theta),np.cos(theta)])
        dx,dz = wrist_pos-np.asarray(c['arm_base'])
        cosine = (dx*dx+dz*dz-l1*l1-l2*l2)/(2*l1*l2)
        if abs(cosine)>1:
            raise ValueError('Unreachable Cartesian pose')
        q2 = -np.arccos(cosine)  # fixed elbow branch; no discontinuous branch switching
        q1 = np.arctan2(dx,dz)-np.arctan2(l2*np.sin(q2),l1+l2*np.cos(q2))
        q3 = float(wrap(theta-np.pi-q1-q2))
        q = np.array([q1,q2,q3])
        bounds=np.asarray(c['arm_joint_ranges'])
        if np.any(q < bounds[:,0]) or np.any(q > bounds[:,1]):
            raise ValueError('IK solution exceeds joint limits')
        return q

    def initialize_robot(self):
        c=self.c
        names=['shoulder','elbow','arm_wrist','left','right','ox','oz','angle']
        self.j={n:int(self.model.joint(n).qposadr[0]) for n in names}
        self.v={n:int(self.model.joint(n).dofadr[0]) for n in names}
        self.arm_q=np.array([self.j[n] for n in names[:3]])
        self.arm_v=np.array([self.v[n] for n in names[:3]])
        self.tcp_id=self.model.site('tcp').id
        target=[self.scene['object_x']+c['block_length']/2+c['initial_approach_gap'],
                self.scene['pedestal_height']+c['block_thickness']/2]
        q=self.inverse_kinematics(target,c['initial_wrist_angle'])
        self.data.qpos[self.arm_q]=q
        self.data.qpos[[self.j['left'],self.j['right']]]=c['max_opening']/2
        self.data.ctrl[:]=[*q,c['max_opening']/2,c['max_opening']/2]
        self.joint_targets=q.copy()
        self.ik_scale=1.
        self.scene['robot']='arm3r'

    @property
    def grip(self):
        return self.data.site_xpos[self.tcp_id][[0,2]].copy()

    @property
    def wrist(self):
        return float(wrap(np.sum(self.data.qpos[self.arm_q])+np.pi))

    @property
    def grip_velocity(self):
        jac=np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self.data,jac,None,self.tcp_id)
        return (jac@self.data.qvel)[[0,2]]

    @property
    def wrist_velocity(self):
        return float(np.sum(self.data.qvel[self.arm_v]))

    def extra_observations(self):
        return {'arm': [*self.data.qpos[self.arm_q],*self.data.qvel[self.arm_v]]}

    def apply_control(self,target,theta,opening):
        self.joint_targets=self.data.qpos[self.arm_q].copy()
        self.ik_scale=0.
        for backtrack in range(self.c['arm_ik_backtracks']):
            scale=0.5**backtrack
            try:
                q=self.inverse_kinematics(self.grip+scale*(target-self.grip),self.wrist+scale*wrap(theta-self.wrist))
            except ValueError:
                continue
            delta=q-self.data.qpos[self.arm_q]
            # Bound joint-space displacement without changing individual joint proportions.
            joint_scale=min(1.,self.c['arm_max_joint_delta']/max(np.max(np.abs(delta)),1e-12))
            self.joint_targets=self.data.qpos[self.arm_q]+joint_scale*delta
            self.ik_scale=scale*joint_scale
            break
        self.data.ctrl[:3]=self.joint_targets
        self.set_gripper_target(opening)

    def advance_physics(self):
        for _ in range(self.substeps):
            self.advance_gripper()
            target=self.joint_targets.copy()
            if self.c['arm_gravity_compensation']:
                # Feedforward is delivered through the same finite-torque actuators.
                target += self.data.qfrc_bias[self.arm_v]/np.asarray(self.c['arm_kp'])
            bounds=np.asarray(self.c['arm_joint_ranges'])
            self.data.ctrl[:3]=np.clip(target,bounds[:,0],bounds[:,1])
            mujoco.mj_step(self.model,self.data)
        # mj_step leaves site transforms at the pre-integration pose. Refresh them
        # before the next Cartesian command and before saving a resumable state.
        mujoco.mj_forward(self.model,self.data)

    def info(self,reason):
        info=super().info(reason)
        info.update(robot='arm3r',ik_scale=float(self.ik_scale),joint_angles=self.data.qpos[self.arm_q].tolist())
        return info

    def snapshot(self):
        state=super().snapshot()
        state['arm_controller']={'joint_targets':self.joint_targets.tolist(),'ik_scale':self.ik_scale}
        return state

    def restore(self,state):
        super().restore(state)
        self.joint_targets=np.asarray(state['arm_controller']['joint_targets'])
        self.ik_scale=state['arm_controller']['ik_scale']
        return self.obs()

    def render(self):
        if self.renderer is None:
            self.renderer=mujoco.Renderer(self.model,height=self.c['render_height'],width=self.c['render_width'])
        self.renderer.update_scene(self.data,camera='front')
        return self.renderer.render()
