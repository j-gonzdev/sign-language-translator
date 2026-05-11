# backend/src/services/video.py
import tempfile
import os
import cv2
import numpy as np
from fastapi import HTTPException

from src.core.config import get_settings

settings = get_settings()


class VideoService:

    @staticmethod
    def validate_and_extract_frames(contenido: bytes, nombre_original: str) -> list[np.ndarray]:
        """
        Valida el vídeo (formato, duración) y extrae frames submuestreados.
        Lanza HTTPException 422 si el vídeo es inválido o supera VIDEO_MAX_DURATION.
        Devuelve lista de frames como arrays numpy BGR.
        """
        # Escribir a fichero temporal — cv2.VideoCapture no acepta bytes en memoria
        suffix = os.path.splitext(nombre_original)[-1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contenido)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise HTTPException(
                    status_code=422,
                    detail="El archivo no es un vídeo válido o está corrupto."
                )

            fps_original = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if fps_original <= 0:
                cap.release()
                raise HTTPException(
                    status_code=422,
                    detail="No se pudo determinar el FPS del vídeo."
                )

            duracion = total_frames / fps_original
            if duracion > settings.VIDEO_MAX_DURATION:
                cap.release()
                raise HTTPException(
                    status_code=422,
                    detail=f"El vídeo supera el límite de {settings.VIDEO_MAX_DURATION} segundos ({duracion:.1f}s)."
                )

            # Submuestreo: coger 1 frame cada N frames para aproximar VIDEO_FPS_SAMPLE
            intervalo = max(1, round(fps_original / settings.VIDEO_FPS_SAMPLE))

            frames = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % intervalo == 0:
                    frames.append(frame)
                frame_idx += 1

            cap.release()
        finally:
            os.unlink(tmp_path)

        if not frames:
            raise HTTPException(
                status_code=422,
                detail="No se pudieron extraer frames del vídeo."
            )

        return frames