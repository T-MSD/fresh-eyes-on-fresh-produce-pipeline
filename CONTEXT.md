# Fresh eyes on fresh produce

Shelf monitoring for the Sensei challenge: for every camera frame and every planogram area, say whether the right product is there and how much of it is left, and turn that into work for store staff.

## Language

### Dataset

**Dataset**:
One load of a planogram and its frames. Every id below is unique within one dataset.
_Avoid_: import, snapshot

**Camera**:
A ceiling camera, keyed by its id (e.g. `6215`). One camera sees one or more gondola modules.

**Frame**:
One image from a camera, named `<unix_ts>.jpg`.
_Avoid_: image, picture, photo

**Planogram**:
`dataset/planogram.json`: which product should occupy each area at the start of the day.

**Area**:
One planogram slot, `area_id` = `<camera>_<n>`, bounded by four corners (top left, top right, bottom right, bottom left) in frame pixels. It may extend past the frame edge.
_Avoid_: slot, region, cell

**Product**:
An item the planogram assigns to areas. Names ending in ` KG` are sold loose; names with a weight are packaged.

### Shelf

**Compliance**:
Whether the product shown in an area is the one the planogram expects.
_Avoid_: correctness, match

**Stock level**:
How full an area is: `empty`, `half` or `full`, with stock value `0`, `0.5` or `1`.
_Avoid_: fill, quantity, stock percentage

**Area status**:
What an area needs, read from its area prediction: `empty`, then `wrong_product`, then `half`, then `ok`. `unknown` when the area has no area prediction in that frame.
_Avoid_: state, health

### Vision

**Crop**:
An area cut out of a frame and warped into a straight rectangle.

**Reference**:
An unlabelled exemplar crop (early-morning "correct and full", emptiest-of-day "empty") used instead of training labels.

### Prediction

**Area prediction**:
The pipeline's answer for one area in one frame: compliance and stock level, with optional confidences.
_Avoid_: label, result

**Fallback**:
An area prediction made without a model answer, e.g. for an unreadable frame. It always exists so no area is missing.
_Avoid_: default, placeholder

**Prediction set**:
An area prediction for every area of every frame of one dataset. Submissions and runs are made from one.
_Avoid_: predictions file, results

**Submission**:
A prediction set written in the organisers' exact format, for scoring.

### Scoring

**Label**:
A human's answer for one area in one frame. Labels are truth; area predictions are not.
_Avoid_: annotation, ground truth (as a noun for one item)

**Dev labels**:
The hand-made labels, in the organisers' `annotations.json` format, used to score methods locally.

**Scoring rules**:
The organisers' points per area: compliance +1 / −2, stock +1 correct, −2 below truth, −3 above truth, −5 for a missing area.

### Runs and tasks

**Run**:
One prediction set stored for one dataset, with the method that made it. Only completed runs are served.

**Task**:
Work for store staff on one area, opened on the frame where a problem appears and closed on the first later frame where it is gone or replaced. Its kind is `restock` (empty), `move_product` (wrong product) or `top_up` (half).
_Avoid_: alert, ticket, job

## Relationships

- A **Dataset** has **Cameras**; a **Camera** has **Areas** and **Frames**; each **Area** expects one **Product**.
- A **Prediction set** holds one **Area prediction** per **Area** per **Frame**; some are **Fallbacks**.
- A **Run** stores one **Prediction set**; its **Tasks** come from the **Area statuses** of its **Area predictions**.
- **Area predictions** are scored against **Labels** with the **Scoring rules**.
