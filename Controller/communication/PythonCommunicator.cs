// communication/PythonCommunicator.cs
using System;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

public class PythonCommunicator : IDisposable
{
    // CHANGED: Fields are now declared as nullable with '?' to resolve CS8618 warnings.
    // This tells the compiler that it's okay for them to be null initially.
    private TcpClient? _dataClient;
    private TcpClient? _commandClient;
    private NetworkStream? _commandStream;

    // CHANGED: Events are also declared as nullable.
    public event Action<string>? OnDataReceived;
    public event Action<string>? OnLogMessage;

    private readonly CancellationTokenSource _cancellationTokenSource;

    public PythonCommunicator()
    {
        _cancellationTokenSource = new CancellationTokenSource();
    }

    // CHANGED: Removed 'async' keyword and now return Task.CompletedTask to resolve CS1998.
    // This method starts background tasks but doesn't need to await them itself.
    public Task ConnectAsync(string host = "127.0.0.1")
    {
        // Start both connection loops in the background
        _ = ConnectAndManageDataClientAsync(host, 9998, _cancellationTokenSource.Token);
        _ = ConnectAndManageCommandClientAsync(host, 9999, _cancellationTokenSource.Token);
        return Task.CompletedTask;
    }

    private async Task ConnectAndManageDataClientAsync(string host, int port, CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                OnLogMessage?.Invoke("Connecting to Python data server...");
                _dataClient = new TcpClient();
                await _dataClient.ConnectAsync(host, port);
                OnLogMessage?.Invoke("✅ Data client connected.");
                await DataReceiverLoopAsync(_dataClient, token);
            }
            // Use a broader exception catch but handle TaskCanceledException specifically to avoid retrying on clean shutdown.
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception)
            {
                OnLogMessage?.Invoke("❌ Data client disconnected. Retrying in 5s...");
                await Task.Delay(5000, token);
            }
        }
    }

    private async Task ConnectAndManageCommandClientAsync(string host, int port, CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            try
            {
                OnLogMessage?.Invoke("Connecting to Python command server...");
                _commandClient = new TcpClient();
                await _commandClient.ConnectAsync(host, port);
                _commandStream = _commandClient.GetStream();
                OnLogMessage?.Invoke("✅ Command client connected.");
                // Keep the connection alive by waiting for cancellation
                await Task.Delay(-1, token);
            }
            catch (TaskCanceledException) { break; }
            catch (Exception)
            {
                OnLogMessage?.Invoke("❌ Command client disconnected. Retrying in 5s...");
                await Task.Delay(5000, token);
            }
        }
    }

    private async Task DataReceiverLoopAsync(TcpClient client, CancellationToken token)
    {
        using var stream = client.GetStream();
        var lengthBuffer = new byte[4];

        while (!token.IsCancellationRequested && client.Connected)
        {
            try
            {
                int bytesRead = await stream.ReadAsync(lengthBuffer, 0, 4, token);
                if (bytesRead < 4) break;

                if (BitConverter.IsLittleEndian) Array.Reverse(lengthBuffer);
                int messageLength = BitConverter.ToInt32(lengthBuffer, 0);

                if (messageLength <= 0) continue; // Skip keep-alive packets

                var messageBuffer = new byte[messageLength];
                var totalBytesRead = 0;
                while (totalBytesRead < messageLength)
                {
                    bytesRead = await stream.ReadAsync(messageBuffer, totalBytesRead, messageLength - totalBytesRead, token);
                    if (bytesRead == 0) throw new Exception("Socket closed prematurely.");
                    totalBytesRead += bytesRead;
                }
                string jsonData = Encoding.UTF8.GetString(messageBuffer);
                OnDataReceived?.Invoke(jsonData);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch
            {
                // Break the loop on any other communication error. The outer loop will handle reconnecting.
                break;
            }
        }
    }

    // CHANGED: The 'payload' parameter is now nullable 'object?' to resolve CS8625.
    public async Task SendCommandAsync(string commandType, object? payload = null)
    {
        // Null checks are now more robust because the fields are nullable.
        if (_commandClient == null || !_commandClient.Connected || _commandStream == null)
        {
            OnLogMessage?.Invoke("⚠️ Cannot send command. Not connected.");
            return;
        }

        try
        {
            var data = new { type = commandType, payload };
            var settings = new JsonSerializerSettings { ContractResolver = new CamelCasePropertyNamesContractResolver() };
            string jsonData = JsonConvert.SerializeObject(data, settings);
            byte[] jsonBytes = Encoding.UTF8.GetBytes(jsonData);

            int messageLength = jsonBytes.Length;
            byte[] lengthPrefix = BitConverter.GetBytes(messageLength);
            if (BitConverter.IsLittleEndian) Array.Reverse(lengthPrefix);

            await _commandStream.WriteAsync(lengthPrefix, 0, lengthPrefix.Length);
            await _commandStream.WriteAsync(jsonBytes, 0, jsonBytes.Length);
            await _commandStream.FlushAsync();
        }
        catch (Exception ex)
        {
            OnLogMessage?.Invoke($"❌ Failed to send command: {ex.Message}");
        }
    }

    public void Dispose()
    {
        _cancellationTokenSource.Cancel();
        _dataClient?.Close();
        _commandClient?.Close();
        _cancellationTokenSource.Dispose();
    }
}