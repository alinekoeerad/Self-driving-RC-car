package com.example.websocketdeneme


import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.SurfaceHolder
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.websocketdeneme.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity(), SurfaceHolder.Callback {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraHandler: CameraHandler
    private lateinit var webSocketHandler: WebSocketHandler
    private val cameraPermission = Manifest.permission.CAMERA
    private val permissionCode = 101

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val surfaceView = binding.cameraPreview
        val surfaceHolder = surfaceView.holder
        surfaceHolder.addCallback(this)

        webSocketHandler = WebSocketHandler("ws://172.18.133.143:80")
        binding.connect.setOnClickListener {
            webSocketHandler.connect()
        }

        cameraHandler = CameraHandler(surfaceHolder, webSocketHandler)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        checkPermission()
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        cameraHandler.refreshCamera()
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        cameraHandler.releaseCamera()
    }

    private fun checkPermission() {
        if (ContextCompat.checkSelfPermission(
                this,
                cameraPermission
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(cameraPermission), permissionCode)
        } else {
            cameraHandler.startCameraPreview()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionCode) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                cameraHandler.startCameraPreview()
            }
        }
    }
}
