# The rules

The benchmark scores one thing: a policy that drives the arm from what a
camera could see and what a person could say.

**A policy is given**, every 8 steps:

- the three fixed camera images (top, side, iso), 512 × 512 RGB, exactly as
  `cameras.observation` renders them;
- the state: 6 joint angles (radians) and the gripper aperture (0 open, 1 closed);
- one instruction clause, as text.

**It returns** the next 8 steps: 6 joint angles and the gripper aperture for each.

**Never, under any name:**

- give the policy object positions, an object list, or any other scene data;
- have the policy output a 3D point, target, waypoint or end pose for
  something else (inverse kinematics, a planner, a reacher) to act on;
- use an object position or target coordinate in a training loss, except
  the one exception below;
- hand-code part of the behaviour around the policy (the gripper, "home",
  when to stop) so that it looks better than it is.

**The one exception, which must be reported:** each training sample also carries
`goal`, the point the teacher is taking the gripper to. A training loss may
use it to reward progress (a "progress reward"). It is never an input or
output of the policy, and results trained with it must say so, because it
cannot exist once the images come from real cameras.

**Results** are the fraction of clauses that succeed on held-out scenes
(seeds from 9,000,000 up), with the policy driving the arm itself through
`core.run_clauses`. Report the `amp-bench eval` table as printed (the
standard is 300 scenes; say so if you used another number), and say what
the policy was trained on.

If a policy does badly, change the policy or the training, not the task.
