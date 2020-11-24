# onnxruntime-for-yolov5
Using CPU to test yolov5s model.

0. predict.py is the model test code。   

1. pytorch to onnx : pytorch2onnx.py;

2. test onnx mode : onnx_cpu_detec.py



There are some TracerWarning at pytorch2onnx.py:

```

TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!

  if self.grid[i].shape[2:4] != x[i].shape[2:4]:
  
TracerWarning: There are 2 live references to the data region being modified when tracing in-place operator copy_ (possibly due to an assignment). This might cause the trace to be incorrect, because all other views that also reference this data will not reflect this change in the trace! On the other hand, if all other views use the same memory chunk, but are disjoint (e.g. are outputs of torch.split), this might still be safe.

  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
  
TracerWarning: There are 2 live references to the data region being modified when tracing in-place operator copy_ (possibly due to an assignment). This might cause the trace to be incorrect, because all other views that also reference this data will not reflect this change in the trace! On the other hand, if all other views use the same memory chunk, but are disjoint (e.g. are outputs of torch.split), this might still be safe.

  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

```

I add this part in the onnx_cpu_detec.py. 
