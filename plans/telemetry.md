# Continuous Drone Localization with Telemetry Integration

## Overview

Extend the existing HLOC-based localization system to provide continuous position updates using telemetry data (velocities, heading, altitude) between HLOC fixes.

**Key Changes:**
- Increase HLOC localization interval from 5s → 20s
- Integrate telemetry data stream for continuous dead reckoning
- Calibrate telemetry coordinate frame to HLOC world frame after first localization
- Visualize both HLOC and telemetry-based positions seamlessly

## User Requirements Summary

1. Change HLOC interval to 20 seconds (from 5 seconds)
2. First localization must use HLOC to establish reference frame
3. Calibrate telemetry heading/altitude to HLOC frame
4. Use telemetry (velocities, heading, altitude) for continuous localization between HLOC fixes
5. Only process telemetry when timestamp changes (avoid stale data)
6. Visualize both HLOC and telemetry-based positions identically

## Coordinate Systems (Confirmed with User)

**Telemetry:**
- Velocities: NED frame (North-East-Down) - **global/world-fixed**
- Heading: 0° = North, increases clockwise (standard compass)
- Altitude: position_z_m (meters)
- Timestamp: Field exists in telemetry JSON

**HLOC:**
- World frame: Arbitrary frame from sparse reconstruction (typically Z-up)
- Pose: cam_from_world as pycolmap.Rigid3d

**Transformation:**
Since velocities are in NED (global frame), we need a **constant rotation** (heading_offset) to align NED with HLOC world frame, determined during calibration.

## Implementation Plan

### File: `/home/raytoningu/Invoras/Hierarchical-Localization/live_localization.py`

#### 1. Configuration Changes (Line 52)
```python
LOCALIZE_INTERVAL = 20.0  # Changed from 5.0
```

#### 2. Add JSON Import (Line 18)
```python
import json
```

#### 3. Replace LocalizationState with HybridLocalizationState (After Line 169)

Create new state management class that handles:
- HLOC poses (ground truth every 20s)
- Telemetry data (continuous updates)
- Dead reckoning position (integrated from velocities)
- Calibration offsets (heading_offset, altitude_offset)

**Key Methods:**
- `update_hloc_pose(pose_data)`: Store HLOC result, reset DR position, trigger calibration if needed
- `update_telemetry(telemetry_data)`: Check timestamp, update DR via integration
- `_calibrate()`: Compute heading_offset and altitude_offset after first HLOC
- `_update_dead_reckoning(prev_telem, curr_telem)`: Integrate velocities to update position
- `get_latest()`: Return current best pose estimate for visualization

**Calibration Logic:**
```python
# Extract HLOC yaw from rotation matrix
hloc_yaw_rad = arctan2(R[1,0], R[0,0])
hloc_yaw_deg = rad2deg(hloc_yaw_rad)

# Compute offsets
heading_offset_deg = hloc_yaw_deg - telemetry_heading_deg
altitude_offset_m = hloc_z - telemetry_altitude_m
```

**Dead Reckoning Update (NED → HLOC frame):**
```python
# Time delta
dt = (curr_timestamp - prev_timestamp).total_seconds()

# NED velocities (already in global frame)
v_ned = [velocity_x_ms, velocity_y_ms, velocity_z_ms]

# Rotate from NED to HLOC using constant heading_offset
theta = deg2rad(heading_offset_deg)
v_hloc_x = v_ned[0] * cos(theta) - v_ned[1] * sin(theta)
v_hloc_y = v_ned[0] * sin(theta) + v_ned[1] * cos(theta)
v_hloc_z = -v_ned[2]  # Flip sign (NED z-down, HLOC z-up)

# Integrate position (XY from velocities, Z from direct measurement)
dr_position[0] += v_hloc_x * dt
dr_position[1] += v_hloc_y * dt
dr_position[2] = telemetry_altitude_m + altitude_offset_m

# Update orientation (yaw only, assume level flight)
yaw = deg2rad(telemetry_heading_deg + heading_offset_deg)
dr_orientation = rotation_matrix_from_yaw(yaw)
```

#### 4. Add Telemetry Handler (Inside connect_to_room, After Line 592)

```python
@room.on("data_received")
def on_data_received(data: rtc.DataPacket):
    """Receive and process telemetry data packets"""
    try:
        telemetry = json.loads(data.data.decode('utf-8'))
        self.localization_state.update_telemetry(telemetry)
    except Exception as e:
        logger.error(f"Failed to process telemetry: {e}")
```

#### 5. Update FrameCapture.__init__ (Line 532)

```python
# Change from:
self.localization_state = LocalizationState(self.sparse_model)

# To:
self.localization_state = HybridLocalizationState(self.sparse_model)
```

#### 6. Update Visualization Status (Line 420-426)

Add source and calibration info to status display:
```python
html.P(f"Frame: {latest['frame_num']} | "
       f"Source: {latest['source'].upper()} | "
       f"HLOC Inliers: {latest['inliers']}/{latest['total_matches']} | "
       f"Calibrated: {'✓' if latest['calibrated'] else '✗'}")
```

## Detailed HybridLocalizationState Class Structure

```python
class HybridLocalizationState:
    def __init__(self, sparse_model):
        self._lock = threading.Lock()

        # HLOC state (ground truth)
        self._latest_hloc_pose = None  # pycolmap.Rigid3d
        self._latest_hloc_camera = None
        self._latest_hloc_inliers = None
        self._latest_hloc_matches = None
        self._latest_hloc_timestamp = None
        self._latest_hloc_frame = None
        self._hloc_initialized = False

        # Telemetry state
        self._latest_telemetry = None  # Dict
        self._last_telemetry_timestamp = None  # For staleness check

        # Dead reckoning (updated continuously)
        self._dr_position = None  # [x, y, z] in HLOC world frame
        self._dr_orientation = None  # 3x3 rotation matrix

        # Calibration offsets (computed after first HLOC)
        self._heading_offset_deg = None  # To add to telemetry heading
        self._altitude_offset_m = None   # To add to telemetry altitude
        self._calibrated = False

        # Status
        self._current_pose_source = "waiting"  # "hloc", "telemetry", "waiting"
        self._error_message = None

        self.sparse_model = sparse_model
```

## Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| First HLOC fails | Keep retrying, no DR until first success |
| HLOC fails after calibration | Continue DR (with drift), show warning |
| Telemetry stops | Freeze at last HLOC position, show warning |
| Timestamp unchanged | Skip telemetry update (stale data) |
| Large dt (>1s) | Skip integration (sanity check) |
| Heading wrap (359°→0°) | Handle in calibration: `(offset + 180) % 360 - 180` |

## Testing Checklist

- [ ] HLOC triggers every 20s (not 5s)
- [ ] Telemetry data received and parsed correctly
- [ ] Calibration occurs after first HLOC
- [ ] Position updates continuously between HLOC fixes
- [ ] Visualization shows smooth movement
- [ ] Source label shows "HLOC" vs "TELEMETRY"
- [ ] Stale telemetry (same timestamp) is ignored
- [ ] HLOC failure doesn't crash system
- [ ] Telemetry loss is handled gracefully

## Implementation Sequence

1. **Config**: Change LOCALIZE_INTERVAL to 20.0, add json import
2. **State Management**: Implement HybridLocalizationState class
3. **HLOC Integration**: Update FrameCapture.__init__ to use new state
4. **Telemetry Handler**: Add on_data_received callback
5. **Calibration**: Implement _calibrate() method
6. **Dead Reckoning**: Implement _update_dead_reckoning() method
7. **Visualization**: Update status display to show source/calibration
8. **Testing**: Verify continuous localization and edge cases

## Critical Files

- `/home/raytoningu/Invoras/Hierarchical-Localization/live_localization.py` - All changes go here
- `/home/raytoningu/Invoras/Hierarchical-Localization/capture_frames.py` - Reference for telemetry handling

## Mathematical Summary

**Calibration (after first HLOC):**
```
heading_offset = atan2(R_hloc[1,0], R_hloc[0,0]) - heading_telem
altitude_offset = z_hloc - z_telem
```

**Velocity Transformation (NED → HLOC):**
```
θ = heading_offset (constant, from calibration)
v_hloc = R(θ) * v_ned
where R(θ) = [[cos(θ), -sin(θ), 0],
              [sin(θ),  cos(θ), 0],
              [0,       0,      -1]]  # -1 to flip Z (NED down → HLOC up)
```

**Position Integration:**
```
dt = current_timestamp - previous_timestamp
position_new = position_old + v_hloc * dt  (for XY)
position_z = z_telem + altitude_offset      (direct measurement)
```

**Orientation Update:**
```
yaw = heading_telem + heading_offset
R_dr = [[cos(yaw), -sin(yaw), 0],
        [sin(yaw),  cos(yaw), 0],
        [0,         0,        1]]
```

## Notes

- Velocities are in **NED (global frame)**, not body frame, so transformation is simpler
- Use **timestamp field** from telemetry JSON to detect new data
- Use **altitude directly** from telemetry (more accurate than integrating velocity_z)
- HLOC resets DR position every 20s to prevent drift accumulation
- Visualization refresh stays at 2s (smooth enough), state updates immediately
