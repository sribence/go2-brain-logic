# follow_executor (port 9113)

Forwards the person follower's command (perception `/follow`) to `mc_motion`
`/move`. It is the only link from the follower to the legs. Remove it with
`docker rm -f nero_go2_follow_executor` and the robot is observe-only again.

## Gates (`decide()`)

A `/move` is sent only when all of these hold:

- the executor is enabled (a remote override latches it off until `POST /enable`)
- `mc_motion` is armed (the executor never arms; the operator does)
- the follower command has `dry_run: false` (possible only with `PERCEPTION_ALLOW_LIVE=1`)
- the follower state is `TRACKING` and not on `hold`
- the follower result is younger than `MAX_RESULT_AGE_S`

On any failed gate it sends one `/stop` and then nothing, so the `mc_motion`
watchdog (0.5 s) also stops the robot if this process dies. `vx` is never
negative and is clamped to the stage limits.

Each tick it also reads `mc_motion /odom` and posts the robot motion since the
last tick to perception `POST /follow/ego`. This moves the follower gate and
the Kalman tracks into the new robot frame.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `EXEC_MAX_VX` | `0.0` | forward limit (stage 1 = turn only) |
| `EXEC_MAX_VYAW` | `0.4` | turn limit, rad/s |
| `MAX_RESULT_AGE_S` | `0.4` | oldest follower result accepted |
| `RATE_HZ` | `10` | loop rate |
| `PERCEPTION_URL`, `MOTION_URL` | `127.0.0.1:9112`, `:9102` | services |

## Status port

`GET /status` (enabled, moving, reason, last command, counters, events),
`POST /enable`, `POST /disable`.

## Run on the robot

The executor uses only the Python standard library and runs in the existing
`nero_go2/web_dashboard` image, like `mc_motion`. There is no restart policy,
so it never starts on its own after a reboot.

```bash
docker run -d --name nero_go2_follow_executor --network host --restart no \
  -e EXEC_MAX_VX=0.0 -e EXEC_MAX_VYAW=0.6 -e MAX_RESULT_AGE_S=0.8 \
  -v /home/unitree/nero_go2_dev/follow_executor:/exec:ro \
  nero_go2/web_dashboard:latest python /exec/executor.py
```
