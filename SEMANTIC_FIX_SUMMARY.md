# Semantic Segmentation Fix Summary

## Issues Fixed

### 1. **Keyframe Labeling Bug**
**Problem**: Only the first keyframe was getting semantic labels. Subsequent keyframes had all points labeled as background.

**Root Cause**: In `tracker.py`, when a frame became a new keyframe:
- It initialized `global_instance_ids` to all zeros
- It only propagated labels from previous keyframe for matching points
- New objects not visible in previous keyframe remained unlabeled

**Fix**: Modified the keyframe labeling logic to:
- After propagating from previous keyframe, check ALL local semantic segments
- For entirely new segments → assign new global IDs
- For partially labeled segments → extend existing labels to unlabeled parts
- Added comprehensive debug output

### 2. **Pose Object Type Error**
**Problem**: Semantic processing failed with error: `'Sim3' object has no attribute 'numpy'`

**Root Cause**: The pose object (`frame.T_WC`) is a lietorch `Sim3` object, but the semantic cache expected a regular tensor/matrix.

**Fix**: Convert Sim3 to matrix before passing to semantic processor:
- Changed `'pose': frame.T_WC` to `'pose': frame.T_WC.matrix()[0]`
- Fixed in both `tracker.py` and `main.py`

## Expected Behavior After Fix

1. **Frame 1 (Init)**: All visible objects labeled
2. **Frame 2-4**: Semantics processed, labels propagated from keyframes
3. **New Keyframes**: 
   - Existing objects keep their IDs
   - New objects get new global IDs
   - Debug output shows labeling progress

4. **Final PLY Output**:
   - All keyframes contribute labeled points
   - Semantic colors show all discovered objects
   - Background percentage significantly reduced

## Debug Output to Verify Fix

Look for these messages:
```
[DEBUG Tracker] Processing new objects for keyframe 2
[DEBUG Tracker] New object - Label 1 (keyboard): 2341 points with global ID 3
[DEBUG Tracker] Keyframe 2: Added 2 new objects after propagation
[DEBUG eval.py] Keyframe 2: 15234 valid points, 4 non-background labels
```

## Files Modified

1. `mast3r_slam/tracker.py`:
   - Fixed keyframe labeling logic (lines 432-481)
   - Fixed pose conversion (lines 79, 361)
   - Added debug output

2. `main.py`:
   - Fixed pose conversion (lines 314, 422)

3. `mast3r_slam/evaluate.py`:
   - Added debug output for PLY generation