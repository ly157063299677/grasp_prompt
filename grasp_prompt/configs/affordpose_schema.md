# Processed AffordPose Sample Schema

Each processed sample is stored as one `.npz` file. The training code expects
the following arrays.

## Required Fields

| Field | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `object_points` | `[N, 3]` | float32 | Normalized object point coordinates. |
| `object_normals` | `[N, 3]` | float32 | Unit surface normals for object points. |
| `affordance_labels` | `[N]` | int64 | Part-level affordance or semantic labels. |
| `valid_mask` | `[N]` | bool/float32 | Task-specific valid prompt region mask. |
| `hand_pose` | `[P]` | float32 | MANO or task-specific hand pose vector. |
| `hand_joints` | `[J, 3]` | float32 | Hand joint locations. |
| `hand_global` | `[G]` | float32 | Global hand or wrist pose vector. |
| `hand_object_state` | `[D]` | float32 | Relative hand-object spatial state. |
| `task_id` | scalar | int64 | Task semantic id. |
| `target_part` | `[R]` | float32 | One-hot target contact-part vector. |
| `target_joint_mask` | `[J]` | float32 | Target joint-chain mask. |
| `prompt_centers` | `[L, 3]` | float32 | Prompt-region centers from contact clustering. |
| `prompt_normals` | `[L, 3]` | float32 | Prompt-region normals. |
| `prompt_part_labels` | `[L]` | int64 | Dominant contact-part label for each region. |
| `prompt_scales` | `[L]` | float32 | Spatial scale for each prompt region. |

## Optional Metadata

| Field | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `object_id` | scalar/string | string | Object instance id for split auditing. |
| `category` | scalar/string | string | Object category for reporting. |

## Study Mappings

Keep the following mappings with the processed data or reference them from the
experiment configuration. They define the exact label space used by a run:

- task id to valid affordance label set;
- MANO/object hand-vertex groups for thumb, index, middle, ring, little, palm,
  and wrist-related target regions;
- contact-part-to-joint association matrix with shape `[R, J]`;
- object-instance split files for in-distribution and unseen-object protocols.
