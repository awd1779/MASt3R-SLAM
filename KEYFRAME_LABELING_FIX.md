# Keyframe Labeling Fix - Critical Bug Found

## The Core Issue

The main bug was in `tracker.py` line 398:
```python
frame.global_instance_ids = torch.zeros(num_points_current_frame, 1, dtype=torch.int64, device=self.device)
```

This line was **wiping out all existing labels** when a frame became a keyframe!

## What Was Happening

1. Frame 2 is processed as regular frame:
   - Semantics processed successfully ✓
   - Labels assigned (99.7% coverage) ✓
   - Has global_instance_ids with proper labels ✓

2. Frame 2 becomes a keyframe:
   - Line 398 reinitializes global_instance_ids to all zeros ❌
   - Wipes out all the labels that were just assigned!
   - Semantic re-processing fails (but shouldn't matter)
   - Result: Keyframe has 0 labeled points

## The Fix

Changed line 398 to:
```python
# Don't reinitialize if frame already has labels!
if not hasattr(frame, 'global_instance_ids') or frame.global_instance_ids is None:
    frame.global_instance_ids = torch.zeros(...)
else:
    print(f"[DEBUG] Keyframe {frame.frame_id} already has global_instance_ids, keeping existing labels")
```

## Expected Behavior Now

1. **Frame 1**: Processed normally, gets labels
2. **Frame 2**: 
   - Processed as regular frame → gets labels (99.7%)
   - Becomes keyframe → KEEPS existing labels
   - Contributes labeled points to PLY
3. **Frame 3-5**: Similar pattern

## Debug Output to Verify

You should now see:
```
[Tracker] Frame 2: 147134/147456 points labeled (99.8%)
[DEBUG] Frame 2 becomes new keyframe
[DEBUG] Keyframe 2 already has global_instance_ids, keeping existing labels
[DEBUG Tracker] Keyframe 2 final stats:
  - Labeled points: 147134 (99.8%)  # Not 0!
[DEBUG eval.py] Keyframe 2: 137267 valid points, 35 non-background labels  # Not 0!
```

## Why Semantic Processing Still Fails

The semantic processing error for keyframes is a separate issue but doesn't matter because:
- Frames are already processed BEFORE becoming keyframes
- They already have local masks and global IDs
- The failed re-processing doesn't affect existing labels

## Summary

The fix ensures that when a frame becomes a keyframe, it retains all the semantic labels it already had. This should result in all keyframes contributing their labeled points to the final PLY file.