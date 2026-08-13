FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY services/vlm_guidance/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY services /app/services
COPY src/__init__.py /app/src/__init__.py
COPY src/vlm_data_generator.py /app/src/vlm_data_generator.py
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "services.vlm_guidance.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
