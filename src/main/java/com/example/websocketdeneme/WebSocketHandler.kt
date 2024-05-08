package com.example.websocketdeneme

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okio.ByteString

class WebSocketHandler(private val url: String) {
    private val client: OkHttpClient = OkHttpClient()
    private var webSocket: WebSocket? = null

    fun connect() {
        Log.d("PieSocket", "Connecting")
        val request: Request = Request
            .Builder()
            .url(url)
            .build()
        val listener = PieSocketListener()
        webSocket = client.newWebSocket(request, listener)
    }

    fun sendVideoData(data: ByteArray) {
        webSocket?.send(ByteString.of(*data))
    }
}

//class WebSocketHandler(private val url: String) {
//    private val client: OkHttpClient = OkHttpClient()
//
//    fun connect() {
//        Log.d("PieSocket", "Connecting")
//        val request: Request = Request
//            .Builder()
//            .url(url)
//            .build()
//        val listener = PieSocketListener()
//        val ws: WebSocket = client.newWebSocket(request, listener)
//    }
//}

