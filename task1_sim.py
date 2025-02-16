import threading
import queue
import time
import numpy as np
import torch
import torch.optim as optim
import math
import matplotlib
# Choose your backend
matplotlib.use("TkAgg")   # or use "Qt5Agg" if you prefer
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# Import your custom modules
from codec.protocolStack import ProtocolStack
from channels.baseChannels import BinarySymmetricChannel
from codec import errorCodecs
from codec.privacyAmp import PrivacyAmplification
from estimators.miEstimator import *

# Global variable for Hash Then Encode toggle
hash_then_encode_enabled = True
# Global reference to the codec instance (so the check button can update it)
global_codec_bob = None

def simulate_bsc_with_codec(bits, channel, codec):
    """
    Simulate one packet transmission through a Binary Symmetric Channel (BSC) with a codec.
    Vectorized for efficiency.
    """
    encoded_bits = codec.encode(bits)
    received_bits = channel.transmit(encoded_bits)
    decoded_bits = codec.decode(received_bits)
    
    # Compute bit error rate and packet error (packet error is 1 if any bit error occurs)
    ber = np.mean(bits != decoded_bits)
    packet_error = np.uint8(ber > 0)
    return ber, packet_error, decoded_bits, encoded_bits

def moving_average(data, window_size=50):
    """Compute the moving average of a 1D list or array."""
    if len(data) < window_size:
        return None
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def simulation_worker(data_queue):
    """
    Run the simulation in a worker thread.
    Updated metrics are sent to the main thread via data_queue.
    """
    global hash_then_encode_enabled, global_codec_bob

    # --- Simulation parameters ---
    num_blocks = 50000  # (not used explicitly below)
    n, k = 511, 40
    t = 95
    mu, fish = k // 2, k  # mu: input bit vector length; fish: PrivacyAmplification parameter

    p_bob = 0.1
    p_eve = 0.2

    batch_size = 64
    hidden_size = math.ceil(1024)
    learning_rate = 0.0005 * 2
    training_steps = 40000
    estimator = "CLUBSample"

    # --- Initialize channels and codecs ---
    channel_bob = BinarySymmetricChannel(p_bob)
    channel_eve = BinarySymmetricChannel(p_eve)
    
    codec_bob = ProtocolStack(codecs=[
        PrivacyAmplification(k=mu, q=fish, enable=hash_then_encode_enabled),
        errorCodecs.BCHCodec(data_block_size=k, code_block_size=n, correction_power=t),
    ])
    # For this simulation, Bob and Eve use the same codec instance.
    codec_eve = codec_bob
    # Make the codec available globally so the GUI can update its flag.
    global_codec_bob = codec_bob

    # --- Initialize MI estimators and optimizers (move to GPU) ---
    mi_estimator_bob = eval(estimator)(mu, mu, hidden_size).cuda()
    optimizer_bob = optim.Adam(mi_estimator_bob.parameters(), learning_rate, amsgrad=True)

    mi_estimator_eve = eval(estimator)(mu, mu, hidden_size).cuda()
    optimizer_eve = optim.Adam(mi_estimator_eve.parameters(), learning_rate, amsgrad=True)

    # --- Lists to store metrics for live plotting ---
    x_data = []
    ber_bob_data = []
    ber_eve_data = []
    packet_error_bob_data = []
    packet_error_eve_data = []
    mi_bob_data = []
    mi_eve_data = []

    total_sent_bits = 0
    window_size = 50  # for moving average (used on the main thread as well)

    # --- Simulation loop ---
    for i in range(training_steps):
        # Check the global flag and update the Hash Then Encode setting
        codec_bob.codecs[0].enable = hash_then_encode_enabled
        
        # Generate a batch of random input bits
        input_bits = np.random.randint(0, 2, size=(batch_size, mu), dtype=np.uint8)
        
        # Accumulators for batch metrics
        ber_bob_batch, ber_eve_batch = [], []
        pkt_err_bob_batch, pkt_err_eve_batch = [], []
        decoded_bits_bob_list, decoded_bits_eve_list = [], []
        
        # Process each sample in the batch
        for j in range(batch_size):
            ber_bob, pkt_err_bob, decoded_bits_bob, _ = simulate_bsc_with_codec(input_bits[j], channel_bob, codec_bob)
            ber_eve, pkt_err_eve, decoded_bits_eve, _ = simulate_bsc_with_codec(input_bits[j], channel_eve, codec_eve)
            
            ber_bob_batch.append(ber_bob)
            pkt_err_bob_batch.append(pkt_err_bob)
            ber_eve_batch.append(ber_eve)
            pkt_err_eve_batch.append(pkt_err_eve)
            decoded_bits_bob_list.append(decoded_bits_bob)
            decoded_bits_eve_list.append(decoded_bits_eve)
        
        # Compute average metrics over the batch
        avg_ber_bob = np.mean(ber_bob_batch)
        avg_pkt_err_bob = np.mean(pkt_err_bob_batch)
        avg_ber_eve = np.mean(ber_eve_batch)
        avg_pkt_err_eve = np.mean(pkt_err_eve_batch)
        
        # Convert to arrays for MI estimation
        decoded_bits_bob_batch = np.array(decoded_bits_bob_list, dtype=np.uint8)
        decoded_bits_eve_batch = np.array(decoded_bits_eve_list, dtype=np.uint8)
        
        # Convert to Torch tensors (centering values around zero)
        input_bits_batch_torch = torch.from_numpy(input_bits).float().cuda() - 0.5
        decoded_bits_bob_batch_torch = torch.from_numpy(decoded_bits_bob_batch).float().cuda() - 0.5
        decoded_bits_eve_batch_torch = torch.from_numpy(decoded_bits_eve_batch).float().cuda() - 0.5
        
        # --- Mutual Information Estimation and Training ---
        # Bob's MI estimation and training
        mi_estimator_bob.eval()
        mi_bob = mi_estimator_bob(input_bits_batch_torch, decoded_bits_bob_batch_torch).item() * np.log2(np.e)
        mi_estimator_bob.train()
        loss_bob = mi_estimator_bob.learning_loss(input_bits_batch_torch, decoded_bits_bob_batch_torch)
        optimizer_bob.zero_grad()
        loss_bob.backward()
        optimizer_bob.step()
        
        # Eve's MI estimation and training
        mi_estimator_eve.eval()
        mi_eve = mi_estimator_eve(input_bits_batch_torch, decoded_bits_eve_batch_torch).item() * np.log2(np.e)
        mi_estimator_eve.train()
        loss_eve = mi_estimator_eve.learning_loss(input_bits_batch_torch, decoded_bits_eve_batch_torch)
        optimizer_eve.zero_grad()
        loss_eve.backward()
        optimizer_eve.step()
        
        # Update cumulative sent bits (each sample contributes mu bits)
        total_sent_bits += batch_size * mu
        x_data.append(total_sent_bits)
        ber_bob_data.append(avg_ber_bob)
        ber_eve_data.append(avg_ber_eve)
        packet_error_bob_data.append(avg_pkt_err_bob)
        packet_error_eve_data.append(avg_pkt_err_eve)
        mi_bob_data.append(mi_bob)
        mi_eve_data.append(mi_eve)
        
        # Clear GPU tensors to manage memory
        del input_bits_batch_torch, decoded_bits_bob_batch_torch, decoded_bits_eve_batch_torch
        torch.cuda.empty_cache()
        
        # Every 100 iterations, send the updated metrics to the main thread
        if i % 5 == 0:
            data_queue.put({
                'x_data': x_data.copy(),
                'ber_bob_data': ber_bob_data.copy(),
                'ber_eve_data': ber_eve_data.copy(),
                'packet_error_bob_data': packet_error_bob_data.copy(),
                'packet_error_eve_data': packet_error_eve_data.copy(),
                'mi_bob_data': mi_bob_data.copy(),
                'mi_eve_data': mi_eve_data.copy(),
                'step': i,
                'mi_bob': mi_bob,
                'mi_eve': mi_eve
            })
            if i > 50:
                print(f"Step {i}/{training_steps} | MI Bob: {np.mean(mi_bob_data[-50:]):.4f} | MI Eve: {np.mean(mi_eve_data[-50:]):.4f}")

    # Signal to the main thread that the simulation is complete
    data_queue.put('DONE')

def main():
    # Create a thread-safe queue for communication between threads
    data_queue = queue.Queue()

    # --- Set up live plotting in the main thread ---
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    plt.subplots_adjust(right=0.75, top=0.9)

    # --- Add CheckButtons widget for Hash Then Encode ---
    # Placed in its own axes on the figure.
    ax_checkbox = fig.add_axes([0.0, 0.0, 0.2, 0.2])  # [left, bottom, width, height]
    check = CheckButtons(ax_checkbox, ['Hash Then Encode'], [hash_then_encode_enabled])
    
    def toggle_hash_then_encode(label):
        global hash_then_encode_enabled, global_codec_bob
        # Update the global flag based on the check button status.
        hash_then_encode_enabled = check.get_status()[0]
        # If the simulation has already created the codec, update its flag.
        if global_codec_bob is not None:
            global_codec_bob.codecs[0].enable = hash_then_encode_enabled
        print("Hash Then Encode Enabled" if hash_then_encode_enabled else "Hash Then Encode Disabled")
    
    check.on_clicked(toggle_hash_then_encode)
    
    # Start the simulation in a separate worker thread.
    sim_thread = threading.Thread(target=simulation_worker, args=(data_queue,))
    sim_thread.start()
    
    window_size = 50  # for moving average
    simulation_running = True

    # --- Main loop: update the plot when new data is received ---
    while simulation_running:
        try:
            # Try to get data from the queue (with a timeout to keep GUI responsive)
            message = data_queue.get(timeout=0.1)
            if message == 'DONE':
                simulation_running = False
                continue

            # Unpack the data dictionary.
            x_data = message['x_data']
            ber_bob_data = message['ber_bob_data']
            ber_eve_data = message['ber_eve_data']
            packet_error_bob_data = message['packet_error_bob_data']
            packet_error_eve_data = message['packet_error_eve_data']
            mi_bob_data = message['mi_bob_data']
            mi_eve_data = message['mi_eve_data']
            step = message['step']
            mi_bob = message['mi_bob']
            mi_eve = message['mi_eve']

            # Clear each subplot.
            for ax in axs:
                ax.cla()

            ber_bob_data = np.array(ber_bob_data)
            ber_eve_data = np.array(ber_eve_data)
            packet_error_bob_data = np.array(packet_error_bob_data)
            packet_error_eve_data = np.array(packet_error_eve_data)

            # Define the threshold to replace small values
            min_value = 1e-5

            # Apply the threshold to avoid zero/negative values in log-scale plots
            ber_bob_data[ber_bob_data <= min_value] = min_value
            ber_eve_data[ber_eve_data <= min_value] = min_value

            packet_error_bob_data[packet_error_bob_data < min_value] = min_value
            packet_error_eve_data[packet_error_eve_data < min_value] = min_value

            # --- Plot BER ---
            axs[0].semilogy(x_data, ber_bob_data, label="BER Bob", color='blue', alpha=0.7)
            axs[0].semilogy(x_data, ber_eve_data, label="BER Eve", color='red', alpha=0.7)
            if len(ber_bob_data) >= window_size:
                ma_ber_bob = moving_average(ber_bob_data, window_size)
                ma_ber_eve = moving_average(ber_eve_data, window_size)
                axs[0].semilogy(x_data[window_size-1:], ma_ber_bob, label="MA BER Bob", linestyle='--', color='blue')
                axs[0].semilogy(x_data[window_size-1:], ma_ber_eve, label="MA BER Eve", linestyle='--', color='red')
            axs[0].set_ylabel("Bit Error Rate")
            axs[0].set_title("BER vs. Sent Bits")
            axs[0].legend()
            axs[0].set_ylim(1e-6, 1) 
            axs[0].set_yscale('log')
            
            # --- Plot Packet Error Rate ---
            axs[1].semilogy(x_data, packet_error_bob_data, label="Packet Error Bob", color='green', alpha=0.7)
            axs[1].semilogy(x_data, packet_error_eve_data, label="Packet Error Eve", color='orange', alpha=0.7)
            if len(packet_error_bob_data) >= window_size:
                ma_pkt_err_bob = moving_average(packet_error_bob_data, window_size)
                ma_pkt_err_eve = moving_average(packet_error_eve_data, window_size)
                axs[1].semilogy(x_data[window_size-1:], ma_pkt_err_bob, label="MA Packet Error Bob", linestyle='--', color='green')
                axs[1].semilogy(x_data[window_size-1:], ma_pkt_err_eve, label="MA Packet Error Eve", linestyle='--', color='orange')
            axs[1].set_ylabel("Packet Error Rate")
            axs[1].set_title("Packet Error Rate vs. Sent Bits")
            axs[1].legend()
            axs[1].set_ylim(1e-6, 1) 
            axs[1].set_yscale('log')

            # --- Plot Mutual Information ---
            axs[2].plot(x_data, mi_bob_data, label="MI Bob", color='purple', alpha=0.7)
            axs[2].plot(x_data, mi_eve_data, label="MI Eve", color='brown', alpha=0.7)
            if len(mi_bob_data) >= window_size:
                ma_mi_bob = moving_average(mi_bob_data, window_size)
                ma_mi_eve = moving_average(mi_eve_data, window_size)
                axs[2].plot(x_data[window_size-1:], ma_mi_bob, label="MA MI Bob", linestyle='--', color='purple')
                axs[2].plot(x_data[window_size-1:], ma_mi_eve, label="MA MI Eve", linestyle='--', color='brown')
            axs[2].set_xlabel("Number of Sent Bits")
            axs[2].set_ylabel("Mutual Information (bits)")
            axs[2].set_title("Mutual Information vs. Sent Bits")
            axs[2].legend()
            
            # --- Use the fourth subplot to display textual progress ---
            axs[3].text(0.5, 0.5, f"Step: {step}\nMI Bob: {mi_bob:.4f}\nMI Eve: {mi_eve:.4f}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=axs[3].transAxes, fontsize=12)
            axs[3].axis('off')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
        except queue.Empty:
            # No new data; keep the GUI responsive.
            plt.pause(0.01)
    
    # Finalize plotting after simulation completes
    plt.ioff()
    plt.show()
    sim_thread.join()

if __name__ == "__main__":
    main()