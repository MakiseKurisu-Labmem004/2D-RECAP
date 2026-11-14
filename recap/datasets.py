"""Dataset names used by generation and evaluation."""

TEST_SPLITS = (
    'id', 'inner_goal_ood', 'outer_goal_ood', 'size_ood',
    'inner_composition_ood', 'outer_composition_ood',
)
DATASET_SPLITS = ('train', 'validation', *TEST_SPLITS)
LEGACY_DATASET_SPLITS = (
    'train', 'validation', 'id', 'goal_ood', 'size_ood', 'composition_ood',
)
