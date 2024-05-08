package com.example.websocketdeneme

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class PermissionManager(private val activity: Activity) {

    private val cameraPermission = Manifest.permission.CAMERA
    private val permissionCode = 101

    fun checkPermission() {
        if (ContextCompat.checkSelfPermission(activity, cameraPermission) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(activity, arrayOf(cameraPermission), permissionCode)
        }
    }

    fun handlePermissionResult(requestCode: Int, grantResults: IntArray, onPermissionGranted: () -> Unit) {
        if (requestCode == permissionCode) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                onPermissionGranted()
            }
        }
    }
}
