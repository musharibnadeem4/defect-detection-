# curl commands

Start the service first (`python tasks.py serve`, or `docker compose up`).
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
curl -s http://127.0.0.1:8000/model-info
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@examples/images/01_defect_correct_cast_def_0_7896.jpeg;type=image/jpeg"
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@examples/images/02_defect_correct_cast_def_0_5311.jpeg;type=image/jpeg"
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@examples/images/03_defect_correct_cast_def_0_3650.jpeg;type=image/jpeg"
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@examples/images/04_defect_correct_cast_def_0_9119.jpeg;type=image/jpeg"
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@examples/images/14_perturbed_blur_sigma2_cast_def_0_7896.png;type=image/png"
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@examples/images/15_perturbed_noise_sigma25_cast_def_0_7896.png;type=image/png"
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@examples/images/16_perturbed_brightness_x1.3_cast_def_0_7896.png;type=image/png"
# errors use one JSON body {"error": {"code", "message", "request_id"}}:
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@README.md;type=text/plain"      # 415
curl -s -X POST http://127.0.0.1:8000/predict -F "file=@README.md;type=image/jpeg"      # 400 (not an image)
```
