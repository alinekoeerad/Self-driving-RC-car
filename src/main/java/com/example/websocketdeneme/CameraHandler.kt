package com.example.websocketdeneme

import android.hardware.Camera
import android.view.SurfaceHolder

class CameraHandler(
    private val surfaceHolder: SurfaceHolder,
    private val webSocketHandler: WebSocketHandler
) {
    private var camera: Camera? = null

    fun startCameraPreview() {
        camera = Camera.open()
        camera?.setPreviewDisplay(surfaceHolder)
        camera?.setPreviewCallback { data, _ ->
            // data is the byte array of the camera preview
            // send this data to the server
            webSocketHandler.sendVideoData(data)
        }
        camera?.startPreview()
    }

//    fun startCameraPreview() {
//        camera = Camera.open()
//        camera?.setPreviewDisplay(surfaceHolder)
//        camera?.startPreview()
//    }

    fun refreshCamera() {
        if (surfaceHolder.surface == null) {
            return
        }
        camera?.stopPreview()
        try {
            camera?.setPreviewDisplay(surfaceHolder)
            camera?.startPreview()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun releaseCamera() {
        camera?.stopPreview()
        camera?.release()
        camera = null
    }
}
