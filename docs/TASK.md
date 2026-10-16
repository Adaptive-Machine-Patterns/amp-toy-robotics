# The task

This is the fixed part of the benchmark. Changing anything here changes
every reported number, so it only changes with a new major version.

## The robot and the scene

- A 6-joint arm on a post, with a two-finger gripper. The joint limits are in
  `core.JOINT_LIMITS`. Two joints (wrist twist, tool twist) do not move the
  gripper and cannot be seen.
- A scene comes from an integer seed: a random start pose and 4 objects, each
  a colour (8) and a shape (5), floating at x, y in ±80 and z in 20..80.
- "Home" is one fixed pose, the same in every scene: the pose scene 666
  starts in (`core.HOME_Q`).

## What a policy is given

Every 8 steps:

1. **Three camera images**, 512 × 512 RGB: top, side and isometric. The
   cameras are fixed (angle, zoom, framing). They show the arm, the gripper
   at its real opening, and the objects. Nothing is drawn in to help.
2. **The state**: 6 joint angles (radians) + gripper opening (0 open, 1 closed).
3. **One instruction clause**, as text.

Nothing else: no object positions, no object list, no target.

## What a policy returns

The next 8 steps: for each, 6 joint angles + gripper opening.

## Instructions

    pick up the X
    move to the X
    move to the midpoint between X and Y
    open the gripper
    close the gripper
    return home

X and Y are "{colour} {shape}". A longer instruction is split on "then" and
run one clause at a time; the policy only ever sees one clause, in canonical
wording (`core.parse_instruction`).

## How a clause runs

`core.run_clauses` shows the policy the current view and state, executes the
8 steps it returns, and repeats. The clause ends when the policy asks for no
further motion (every joint within 0.03 rad and the gripper within 0.06 of
where it is), or after 200 steps (120 before version 0.3.0). The policy decides everything about the
motion, including the gripper and when to stop.

## The world

- **Joint speed limit (from 0.2.0):** no joint moves more than 0.12 rad in one
  step. A step that asks for more is scaled down as a whole, so the arm keeps
  its direction but takes longer. The teacher never reaches the limit. It keeps
  policies that learn to cut corners (for example to shorten their path) to
  speeds a real arm could manage. `evaluate` reports the share of steps it
  slowed.
- An object is picked up only if the gripper closes (opening passes 0.9)
  with its grasp point within 8 units of the object.
- A held object moves with the gripper. It stays where the gripper is when
  the gripper opens (below 0.5).

## Success

| clause | succeeds if |
|---|---|
| pick up X | X is in the gripper afterwards |
| move to … | the grasp point ends within 8 units of the target |
| open / close | the gripper ends in that state, and no joint moved more than 0.05 rad |
| return home | every visible joint ends within 0.05 rad of the home pose |

A clause that hits the 200-step limit fails. The limit is a safety stop for a
policy that never says it is done, not a race: it is about 3.5 times the
teacher's longest clauses. Raising it from 120 (in 0.3.0) can only turn
failures into successes, since policies are deterministic, so scores
measured under 0.1.0 or 0.2.0 are lower bounds.

## The teacher

`core.teacher_step` is a hand-written controller that solves the task using
the true object positions. It is not a learned model and not a valid
entry: it defines what "doing it right" looks like and labels the training
data (`data.py`). It scores 100% on the standard 300 held-out scenes. Asked to
pick up one object while holding another, it first sets the held one down
clear of the target (from 0.3.0; before, it dropped it on the target, grabbed
it again, and looped until the step limit in 0.5% of clauses). Its
action is a fixed function of the current state, so it can label any
state, including ones a learner wanders into.

## Seeds

| seeds | use |
|---|---|
| 0 .. 8,999,999 | training (`data.stream`, `data.samples_for_seed`) |
| 9,000,000 and up | held out: never train on these |
| 9,010,000 .. 9,010,299 | closed-loop evaluation: the standard 300 scenes (`amp-bench eval`) |
| 9,050,000 .. | grounding tests (default) |
