import asyncio
import websockets

async def server(websocket, path):
    # When a client connects, this function will be called
    # `websocket` is the WebSocket connection object, `path` is the URL path requested
    print("Client connected")
    while True:
        # Wait for data from the client
        message = await websocket.recv()
        print(f"{message}")
        
        # Send a response back to the client
        response = "Received"
        await websocket.send(response)
        print(f"Sent response: {response}")

async def main():
    # Start the WebSocket server on the specified IP address and port
    async with websockets.serve(server, "172.18.133.143", 80, max_size=None):
        print("WebSocket server started")
        # Keep the server running indefinitely
        await asyncio.Future()

# Run the main coroutine
asyncio.run(main())

