import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from tqdm import tqdm
import warnings
import os # Added for os.path.exists

warnings.filterwarnings('ignore')

class BoltzmannSpikingNeuralNetwork:
    def __init__(self, input_size=784, hidden_size=500, output_size=10):
        """
        Spiking Sinir Ağı (SNN) ve Boltzmann Makinesi entegrasyonu ile bir sinir ağı modeli.
        Bu sınıf, sızıntılı integrate-and-fire (LIF) nöron modelini,
        spike zamanlamasına bağlı plastisite (STDP) öğrenme kuralını ve
        bir Boltzmann makinesi katmanını içerir.
        """
        # Ağ yapısı
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # LIF Parametreleri
        self.V_rest = -70.0      # mV (Dinlenme potansiyeli)
        self.V_reset = -80.0     # mV (Düşük reset potansiyeli)
        self.V_threshold = -55.0 # mV (Düşük eşik potansiyeli)
        self.R = 10.0            # MΩ (Membran direnci)
        self.tau_m = 10.0        # ms (Membran zaman sabiti)
        self.dT = 0.5            # ms (Simülasyon zaman adımı)
        
        # Adaptif eşik parametreleri
        self.V_th_adapt = np.full(hidden_size + output_size, self.V_threshold) 
        self.tau_th = 30.0       # ms (Eşik adaptasyon zaman sabiti)
        self.alpha_th = 1.5      # mV (Eşik adaptasyon artış miktarı)
        
        # STDP Parametreleri
        self.A_plus = 0.05       
        self.A_minus = 0.04      
        self.tau_plus = 12.0     
        self.tau_minus = 15.0    
        
        # Multi-scale STDP için parametreler (yavaş bileşenler)
        self.A_plus_slow = 0.01
        self.A_minus_slow = 0.008
        self.tau_plus_slow = 50.0
        self.tau_minus_slow = 60.0
        
        # Xavier/He başlangıç değerleri
        self.W_input_hidden = self._he_init((hidden_size, input_size)) 
        self.W_hidden_output = self._he_init((output_size, hidden_size)) 
        self.W_hidden_hidden = self._xavier_init((hidden_size, hidden_size)) * 0.5 
        np.fill_diagonal(self.W_hidden_hidden, 0)
        
        # Boltzmann Makinesi ağırlıkları
        self.W_boltzmann = self._xavier_init((hidden_size, hidden_size)) * 0.3 
        np.fill_diagonal(self.W_boltzmann, 0)
        self.temperature = 1.5 
        
        # İnhibisyon matrisi
        self.W_inhibition = np.random.uniform(0.2, 0.6, (output_size, output_size)) 
        np.fill_diagonal(self.W_inhibition, 0)
        
        # Homeostatik parametreler
        self.target_rates = np.full(hidden_size, 5.0)  # Hz (Hedef ateşleme oranları)
        self.scaling_factors = np.ones(hidden_size)    # Ölçekleme faktörleri
        self.homeostatic_rate = 0.005                  # Homeostatik öğrenme oranı
        
        # Üstel azalma ön hesaplamaları
        self.exp_decay_plus = np.exp(-self.dT / self.tau_plus)
        self.exp_decay_minus = np.exp(-self.dT / self.tau_minus)
        self.exp_decay_plus_slow = np.exp(-self.dT / self.tau_plus_slow)
        self.exp_decay_minus_slow = np.exp(-self.dT / self.tau_minus_slow)
        self.exp_decay_th = np.exp(-self.dT / self.tau_th)
        
    def _he_init(self, shape):
        fan_in = shape[1]
        std = np.sqrt(2.0 / fan_in)
        return np.random.normal(0, std, shape).astype(np.float32)
    
    def _xavier_init(self, shape):
        fan_in, fan_out = shape[1], shape[0]
        std = np.sqrt(2.0 / (fan_in + fan_out)) 
        return np.random.normal(0, std, shape).astype(np.float32)
    
    def encode_population_poisson(self, data_batch, duration=100, max_freq=150): 
        batch_size = data_batch.shape[0]
        enhanced_data = self._enhance_contrast(data_batch)
        base_rates = enhanced_data * max_freq
        noise = np.random.normal(0, 0.03 * max_freq, base_rates.shape)
        rates = np.maximum(base_rates + noise, 0.5)
        spike_probs = np.minimum(rates * self.dT / 1000.0, 0.8) 
        random_vals = np.random.random((batch_size, duration, self.input_size))
        spikes = random_vals < spike_probs[:, np.newaxis, :]
        return spikes.astype(np.float32)
    
    def _enhance_contrast(self, data):
        mean_val = np.mean(data, axis=1, keepdims=True)
        std_val = np.std(data, axis=1, keepdims=True) + 1e-6
        normalized = (data - mean_val) / std_val
        enhanced = 1.0 / (1.0 + np.exp(-1.5 * normalized))
        return enhanced.astype(np.float32)
    
    def boltzmann_update(self, hidden_states_batch, learn=True):
        if not learn:
            return hidden_states_batch
        
        local_fields_batch = np.dot(hidden_states_batch, self.W_boltzmann.T) 
        probs_batch = 1.0 / (1.0 + np.exp(-local_fields_batch / self.temperature)) 
        
        new_states_batch = np.random.random(hidden_states_batch.shape) < probs_batch
        return new_states_batch.astype(np.float32)

    def adaptive_lif_dynamics(self, V_batch, I_batch, neuron_indices, is_output_layer=False):
        I_leak = -(V_batch - self.V_rest) / self.tau_m
        dV = I_leak + self.R * I_batch / self.tau_m
        new_V = V_batch + self.dT * dV
        return new_V.astype(np.float32)
    
    def forward_pass(self, input_spikes_batch, learn=True): # Renamed from forward_pass_advanced
        batch_size, duration, _ = input_spikes_batch.shape
        hidden_spikes_batch_t = np.zeros((batch_size, duration, self.hidden_size), dtype=np.float32)
        output_spikes_batch_t = np.zeros((batch_size, duration, self.output_size), dtype=np.float32)
        
        V_hidden_batch = np.full((batch_size, self.hidden_size), self.V_rest, dtype=np.float32)
        V_output_batch = np.full((batch_size, self.output_size), self.V_rest, dtype=np.float32)
        
        if learn:
            pre_trace_input_fast = np.zeros((batch_size, self.input_size), dtype=np.float32)
            post_trace_hidden_fast = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
            pre_trace_hidden_fast = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
            post_trace_output_fast = np.zeros((batch_size, self.output_size), dtype=np.float32)
            
            pre_trace_input_slow = np.zeros((batch_size, self.input_size), dtype=np.float32)
            post_trace_hidden_slow = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
            pre_trace_hidden_slow = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
            post_trace_output_slow = np.zeros((batch_size, self.output_size), dtype=np.float32)

        firing_rates_hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float32)

        current_V_th_hidden = np.tile(self.V_th_adapt[:self.hidden_size], (batch_size, 1)).astype(np.float32)
        current_V_th_output = np.tile(self.V_th_adapt[self.hidden_size:], (batch_size, 1)).astype(np.float32)

        for t in range(duration):
            current_input_spikes = input_spikes_batch[:, t]

            I_syn_hidden = np.dot(current_input_spikes, self.W_input_hidden.T)
            I_syn_hidden *= self.scaling_factors[np.newaxis, :] 
            
            if t > 0:
                I_rec_hidden = np.dot(hidden_spikes_batch_t[:, t-1], self.W_hidden_hidden.T)
                I_syn_hidden += I_rec_hidden
            
            V_hidden_batch = self.adaptive_lif_dynamics(V_hidden_batch, I_syn_hidden, None)
            
            spike_mask_hidden = V_hidden_batch >= current_V_th_hidden
            hidden_spikes = spike_mask_hidden.astype(np.float32)
            
            V_hidden_batch[spike_mask_hidden] = self.V_reset
            if learn:
                current_V_th_hidden[spike_mask_hidden] += self.alpha_th 
            
            hidden_spikes_batch_t[:, t] = hidden_spikes
            firing_rates_hidden += hidden_spikes 

            if learn and t > 0 and t % 10 == 0: 
                hidden_spikes_updated_boltzmann = self.boltzmann_update(hidden_spikes.copy(), learn=True)
                hidden_spikes_batch_t[:, t] = hidden_spikes_updated_boltzmann 
                hidden_spikes = hidden_spikes_updated_boltzmann
            
            I_syn_output = np.dot(hidden_spikes, self.W_hidden_output.T)
            
            if t > 0:
                I_inhib = -np.dot(output_spikes_batch_t[:, t-1], self.W_inhibition.T) * 1.2 
                I_syn_output += I_inhib
            
            V_output_batch = self.adaptive_lif_dynamics(V_output_batch, I_syn_output, None, is_output_layer=True)
            
            spike_mask_output = V_output_batch >= current_V_th_output
            output_spikes = spike_mask_output.astype(np.float32)
            
            V_output_batch[spike_mask_output] = self.V_reset
            if learn:
                current_V_th_output[spike_mask_output] += self.alpha_th

            output_spikes_batch_t[:, t] = output_spikes
            
            if learn: 
                decay_factor_th = self.exp_decay_th
                current_V_th_hidden[~spike_mask_hidden] = self.V_threshold + \
                    (current_V_th_hidden[~spike_mask_hidden] - self.V_threshold) * decay_factor_th
                current_V_th_output[~spike_mask_output] = self.V_threshold + \
                    (current_V_th_output[~spike_mask_output] - self.V_threshold) * decay_factor_th

            if learn and t > 0 : 
                pre_trace_input_fast = pre_trace_input_fast * self.exp_decay_plus + current_input_spikes
                post_trace_hidden_fast = post_trace_hidden_fast * self.exp_decay_minus + hidden_spikes
                pre_trace_hidden_fast = pre_trace_hidden_fast * self.exp_decay_plus + hidden_spikes
                post_trace_output_fast = post_trace_output_fast * self.exp_decay_minus + output_spikes

                pre_trace_input_slow = pre_trace_input_slow * self.exp_decay_plus_slow + current_input_spikes
                post_trace_hidden_slow = post_trace_hidden_slow * self.exp_decay_minus_slow + hidden_spikes
                pre_trace_hidden_slow = pre_trace_hidden_slow * self.exp_decay_plus_slow + hidden_spikes
                post_trace_output_slow = post_trace_output_slow * self.exp_decay_minus_slow + output_spikes

                avg_post_trace_hidden_fast = np.mean(post_trace_hidden_fast, axis=0)
                avg_current_input_spikes = np.mean(current_input_spikes, axis=0)
                avg_pre_trace_input_fast = np.mean(pre_trace_input_fast, axis=0)
                avg_hidden_spikes = np.mean(hidden_spikes, axis=0) 
                avg_output_spikes = np.mean(output_spikes, axis=0)

                dW_pot_ih = self.A_plus * np.outer(avg_post_trace_hidden_fast, avg_current_input_spikes)
                dW_dep_ih = self.A_minus * np.outer(avg_hidden_spikes, avg_pre_trace_input_fast)
                self.W_input_hidden += (dW_pot_ih - dW_dep_ih) * 0.01 

                avg_post_trace_hidden_slow = np.mean(post_trace_hidden_slow, axis=0)
                avg_pre_trace_input_slow = np.mean(pre_trace_input_slow, axis=0)
                dW_pot_ih_slow = self.A_plus_slow * np.outer(avg_post_trace_hidden_slow, avg_current_input_spikes)
                dW_dep_ih_slow = self.A_minus_slow * np.outer(avg_hidden_spikes, avg_pre_trace_input_slow)
                self.W_input_hidden += (dW_pot_ih_slow - dW_dep_ih_slow) * 0.01

                if np.sum(output_spikes) > 0: 
                    avg_post_trace_output_fast = np.mean(post_trace_output_fast, axis=0)
                    avg_pre_trace_hidden_fast = np.mean(pre_trace_hidden_fast, axis=0)
                    
                    dW_pot_ho = self.A_plus * np.outer(avg_post_trace_output_fast, avg_hidden_spikes)
                    dW_dep_ho = self.A_minus * np.outer(avg_output_spikes, avg_pre_trace_hidden_fast)
                    self.W_hidden_output += (dW_pot_ho - dW_dep_ho) * 0.02 

                    avg_post_trace_output_slow = np.mean(post_trace_output_slow, axis=0)
                    avg_pre_trace_hidden_slow = np.mean(pre_trace_hidden_slow, axis=0)
                    dW_pot_ho_slow = self.A_plus_slow * np.outer(avg_post_trace_output_slow, avg_hidden_spikes)
                    dW_dep_ho_slow = self.A_minus_slow * np.outer(avg_output_spikes, avg_pre_trace_hidden_slow)
                    self.W_hidden_output += (dW_pot_ho_slow - dW_dep_ho_slow) * 0.02
                
                if learn and t > 0 and t % 10 == 0 and np.sum(hidden_spikes) > self.hidden_size * 0.05 : 
                    co_activation = np.mean(np.einsum('bi,bj->bij', hidden_spikes, hidden_spikes), axis=0)
                    self.W_boltzmann += 0.0005 * (co_activation - np.mean(co_activation)) 
                    self.W_boltzmann = np.clip(self.W_boltzmann, -0.8, 0.8) 
                    np.fill_diagonal(self.W_boltzmann, 0)

        if learn:
            self.V_th_adapt[:self.hidden_size] = np.mean(current_V_th_hidden, axis=0)
            self.V_th_adapt[self.hidden_size:] = np.mean(current_V_th_output, axis=0)

            avg_firing_batch = np.mean(firing_rates_hidden, axis=0) / (duration * self.dT / 1000.0) 
            scaling_update = self.homeostatic_rate * (self.target_rates - avg_firing_batch)
            self.scaling_factors = np.clip(self.scaling_factors + scaling_update, 0.2, 2.5) 
        
            self.W_input_hidden = np.clip(self.W_input_hidden, 0.0, 2.5) 
            self.W_hidden_output = np.clip(self.W_hidden_output, 0.0, 3.0) 
            self.W_hidden_hidden = np.clip(self.W_hidden_hidden, -0.3, 0.8) 

        return hidden_spikes_batch_t, output_spikes_batch_t
    
    def predict_ensemble(self, input_data_batch, n_runs=3):
        batch_size = input_data_batch.shape[0]
        all_run_predictions = np.full((n_runs, batch_size), -1, dtype=int)
        all_run_confidences = np.zeros((n_runs, batch_size), dtype=np.float32)
        
        for run_idx in range(n_runs): 
            input_spikes_batch = self.encode_population_poisson(input_data_batch, duration=80) 
            _, output_spikes_batch = self.forward_pass(input_spikes_batch, learn=False) # Updated call
            spike_counts_batch = np.sum(output_spikes_batch, axis=1)
            
            total_spikes_per_sample = np.sum(spike_counts_batch, axis=1) + 1e-7
            no_spike_mask = total_spikes_per_sample < (1.0 + 1e-7)
            predictions_run = np.argmax(spike_counts_batch, axis=1)
            confidences_run = spike_counts_batch[np.arange(batch_size), predictions_run] / total_spikes_per_sample

            min_spike_mask = spike_counts_batch[np.arange(batch_size), predictions_run] >= 1
            if np.any(min_spike_mask):
                sorted_spikes = np.sort(spike_counts_batch[min_spike_mask], axis=1)[:, ::-1]
                margin = sorted_spikes[:, 0] - (sorted_spikes[:, 1] if self.output_size > 1 else 0)
                confidences_run[min_spike_mask] += 0.05 * np.tanh(margin / 1.5)
            
            low_activation_pred_mask = spike_counts_batch[np.arange(batch_size), predictions_run] < 1
            confidences_run[low_activation_pred_mask] *= 0.3
            
            predictions_run[no_spike_mask] = -1
            confidences_run[no_spike_mask] = 0.0
            
            all_run_predictions[run_idx] = predictions_run
            all_run_confidences[run_idx] = np.clip(confidences_run, 0.0, 1.0)
        
        final_predictions = np.full(batch_size, -1, dtype=int)
        final_confidences = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            votes = all_run_predictions[:, i]
            confs_for_sample = all_run_confidences[:, i]
            
            valid_votes_mask = votes != -1
            valid_votes = votes[valid_votes_mask]
            valid_confs = confs_for_sample[valid_votes_mask]

            if not len(valid_votes): 
                final_predictions[i] = np.random.randint(0, self.output_size) 
                final_confidences[i] = 0.01
            else:
                unique_votes, vote_indices = np.unique(valid_votes, return_inverse=True)
                weighted_counts = np.zeros_like(unique_votes, dtype=float)
                
                np.add.at(weighted_counts, vote_indices, valid_confs)

                if np.sum(weighted_counts) > 0:
                    best_vote_idx = np.argmax(weighted_counts)
                    final_predictions[i] = unique_votes[best_vote_idx]
                    winning_vote_confs = valid_confs[valid_votes == final_predictions[i]]
                    final_confidences[i] = np.mean(winning_vote_confs) if len(winning_vote_confs) > 0 else 0.0
                else: 
                    counts = np.bincount(valid_votes)
                    final_predictions[i] = np.argmax(counts) if len(counts) > 0 else np.random.randint(0, self.output_size)
                    final_confidences[i] = np.mean(valid_confs) if len(valid_confs) > 0 else 0.0
        
        return np.array(final_predictions), np.clip(final_confidences, 0.0, 1.0)

    def train_epoch_supervised(self, X_train, y_train, batch_size=32, supervised_weight=0.15):
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        correct_total = 0
        total_processed_for_acc = 0
        
        for batch_idx in tqdm(range(n_batches), desc="Eğitim Adımları"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx] 
            current_batch_size = len(X_batch)
            
            if current_batch_size == 0: continue

            input_spikes_batch = self.encode_population_poisson(X_batch, duration=100) 
            hidden_spikes_history_batch, output_spikes_history_batch = self.forward_pass(input_spikes_batch, learn=True) # Updated call
            
            output_neuron_spikes_batch = np.sum(output_spikes_history_batch, axis=1)
            sum_output_spikes_per_sample = np.sum(output_neuron_spikes_batch, axis=1, keepdims=True) + 1e-6

            valid_sample_mask = np.squeeze(sum_output_spikes_per_sample > 1e-5)
            if not np.any(valid_sample_mask):
                continue
                
            active_X_batch = X_batch[valid_sample_mask]
            active_y_batch = y_batch[valid_sample_mask]
            active_output_neuron_spikes = output_neuron_spikes_batch[valid_sample_mask]
            active_sum_output_spikes = sum_output_spikes_per_sample[valid_sample_mask]
            active_hidden_spikes_history = hidden_spikes_history_batch[valid_sample_mask]
            num_active_samples = active_X_batch.shape[0]

            predicted_labels = np.argmax(active_output_neuron_spikes, axis=1)
            target_labels = active_y_batch.astype(int)

            max_spikes_per_sample = np.max(active_output_neuron_spikes, axis=1)
            target_neuron_actual_spikes = active_output_neuron_spikes[np.arange(num_active_samples), target_labels]
            
            needs_update_mask = (predicted_labels != target_labels) | \
                                (target_neuron_actual_spikes < max_spikes_per_sample * 0.8)

            avg_hidden_activation_batch = np.mean(active_hidden_spikes_history, axis=1)
            significant_hidden_activity_mask = np.sum(avg_hidden_activation_batch, axis=1) > 0.01
            
            final_update_mask = needs_update_mask & significant_hidden_activity_mask
            
            if np.any(final_update_mask):
                update_indices = np.where(final_update_mask)[0]

                update_target_labels = target_labels[update_indices]
                update_predicted_labels = predicted_labels[update_indices]
                update_avg_hidden_activation = avg_hidden_activation_batch[update_indices]
                update_output_neuron_spikes = active_output_neuron_spikes[update_indices]
                update_sum_output_spikes = active_sum_output_spikes[update_indices]
                
                num_to_update = len(update_target_labels)

                target_spike_ratio = update_output_neuron_spikes[np.arange(num_to_update), update_target_labels] / np.squeeze(update_sum_output_spikes)
                error_target = 1.0 - target_spike_ratio
                update_magnitude_target = supervised_weight * error_target[:, np.newaxis] 
                dW_target = update_magnitude_target * update_avg_hidden_activation
                
                np.add.at(self.W_hidden_output, update_target_labels, dW_target)

                depress_mask = (update_predicted_labels != update_target_labels) & \
                               (update_output_neuron_spikes[np.arange(num_to_update), update_predicted_labels] > 0)
                
                if np.any(depress_mask):
                    depress_indices_final = np.where(depress_mask)[0]

                    depress_predicted_labels_final = update_predicted_labels[depress_indices_final]
                    depress_avg_hidden_final = update_avg_hidden_activation[depress_indices_final]
                    depress_output_spikes_final = update_output_neuron_spikes[depress_indices_final]
                    depress_sum_output_spikes_final = update_sum_output_spikes[depress_indices_final]
                    num_to_depress = len(depress_predicted_labels_final)

                    wrong_spike_ratio = depress_output_spikes_final[np.arange(num_to_depress), depress_predicted_labels_final] / np.squeeze(depress_sum_output_spikes_final)
                    error_wrong = wrong_spike_ratio
                    update_magnitude_wrong = supervised_weight * 0.5 * error_wrong[:, np.newaxis]
                    dW_wrong = update_magnitude_wrong * depress_avg_hidden_final
                    
                    np.subtract.at(self.W_hidden_output, depress_predicted_labels_final, dW_wrong)
            
            self.W_hidden_output = np.clip(self.W_hidden_output, 0.0, 3.0) 
            
            predictions, _ = self.predict_ensemble(X_batch, n_runs=1) 
            valid_preds_mask = predictions != -1
            num_valid_in_batch = np.sum(valid_preds_mask)

            if num_valid_in_batch > 0:
                correct_batch = np.sum(predictions[valid_preds_mask] == y_batch[valid_preds_mask])
                correct_total += correct_batch
                total_processed_for_acc += num_valid_in_batch
            
            if batch_idx > 0 and batch_idx % 30 == 0: 
                self.temperature = max(0.3, self.temperature * 0.985) 
        
        avg_accuracy = correct_total / total_processed_for_acc if total_processed_for_acc > 0 else 0
        return avg_accuracy, 0.0

def main(): # Renamed from main_advanced
    print("Spiking Sinir Ağı ve Boltzmann Makinesi Uygulaması\n")
    
    train_path = 'train.csv' # Relative path
    test_path = 'test.csv'   # Relative path
        
    print("Veri dosyaları yükleniyor...")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"UYARI: '{train_path}' veya '{test_path}' dosyaları bulunamadı.")
        print("Lütfen dosya yollarını kontrol edin veya ilgili CSV dosyalarını komut dosyasının bulunduğu dizine yerleştirin.")
        print("Örnek veri seti oluşturuluyor (Bu, gerçek MNIST verisi değildir!)...")
        num_train_samples = 1000
        num_test_samples = 200
        img_size = 784 # 28x28
        train_data = np.random.randint(0, 256, size=(num_train_samples, img_size))
        train_labels = np.random.randint(0, 10, size=num_train_samples)
        train_df = pd.DataFrame(train_data)
        train_df['label'] = train_labels
        
        test_data = np.random.randint(0, 256, size=(num_test_samples, img_size))
        test_df = pd.DataFrame(test_data)
        print("Örnek veri seti başarıyla oluşturuldu.")
    else:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print("Veri dosyaları başarıyla yüklendi.")
        except Exception as e:
            print(f"HATA: Veri dosyaları yüklenirken bir sorun oluştu: {e}")
            print("Örnek veri seti oluşturuluyor (Bu, gerçek MNIST verisi değildir!)...")
            num_train_samples = 1000
            num_test_samples = 200
            img_size = 784 # 28x28
            train_data = np.random.randint(0, 256, size=(num_train_samples, img_size))
            train_labels = np.random.randint(0, 10, size=num_train_samples)
            train_df = pd.DataFrame(train_data)
            train_df['label'] = train_labels
            
            test_data = np.random.randint(0, 256, size=(num_test_samples, img_size))
            test_df = pd.DataFrame(test_data)
            print("Örnek veri seti başarıyla oluşturuldu.")


    X_train_all = train_df.drop('label', axis=1).values.astype(np.float32) / 255.0 
    y_train_all = train_df['label'].values.astype(np.int32)
    X_test_all = test_df.values.astype(np.float32) / 255.0 
    
    print(f"Eğitim veri seti boyutu: {X_train_all.shape}, Test veri seti boyutu: {X_test_all.shape}")
    
    subset_size = min(15000, X_train_all.shape[0])
    X_train_subset = X_train_all[:subset_size]
    y_train_subset = y_train_all[:subset_size]
    
    print("Boltzmann SNN modeli oluşturuluyor...")
    snn = BoltzmannSpikingNeuralNetwork(input_size=X_train_all.shape[1], hidden_size=500, output_size=10) 
    
    n_epochs = 12
    batch_size = 64 
    print(f"\n{n_epochs} epoch eğitim süreci başlıyor...")
    
    train_accuracies = []
    
    for epoch in range(n_epochs):
        print(f"\n--- Epoch {epoch + 1}/{n_epochs} ---")
        
        indices = np.random.permutation(len(X_train_subset))
        X_shuffled = X_train_subset[indices]
        y_shuffled = y_train_subset[indices]
        
        sup_weight = 0.1 + 0.15 * (epoch / (n_epochs -1)) if n_epochs > 1 else 0.1 
        
        accuracy, _ = snn.train_epoch_supervised(X_shuffled, y_shuffled, 
                                                 batch_size=batch_size, 
                                                 supervised_weight=sup_weight)
        
        train_accuracies.append(accuracy)
        print(f"Eğitim Doğruluğu: {accuracy:.4f}")
        print(f"Mevcut Sıcaklık: {snn.temperature:.3f}")
        
        if accuracy > 0.90 and epoch > n_epochs // 2:
             print("Yüksek doğruluk oranına ulaşıldı, eğitim erken sonlandırılıyor...")
             break
    
    print("\nTest veri seti üzerinde değerlendirme yapılıyor...")
    test_subset_size = min(3000, X_test_all.shape[0]) 
    X_test_eval_subset = X_test_all[:test_subset_size]

    test_batch_size = 100 
    all_predictions_test = []
    all_confidences_test = []
    
    if len(X_test_eval_subset) > 0:
        for i in tqdm(range(0, len(X_test_eval_subset), test_batch_size), desc="Test Değerlendirmesi"):
            batch_end = min(i + test_batch_size, len(X_test_eval_subset))
            X_batch_test = X_test_eval_subset[i:batch_end]
            if len(X_batch_test) == 0: continue
            batch_preds, batch_confs = snn.predict_ensemble(X_batch_test, n_runs=5) 
            all_predictions_test.extend(batch_preds)
            all_confidences_test.extend(batch_confs)
    else:
        print("Değerlendirilecek test verisi bulunmamaktadır.")

    print(f"\n=== SONUÇLAR ===")
    if train_accuracies:
        print(f"Son Eğitim Doğruluğu: {train_accuracies[-1]:.4f}")
    else:
        print("Eğitim tamamlanmadı veya bir sorun oluştu.")
    
    if all_confidences_test:
        print(f"Ortalama Test Güveni: {np.mean(all_confidences_test):.4f}")
        valid_test_preds_mask = np.array(all_predictions_test) != -1
        num_valid_test_preds = np.sum(valid_test_preds_mask)
        print(f"Geçerli Test Tahmini Sayısı: {num_valid_test_preds} / {len(all_predictions_test)}")
    else:
        print("Test tahminleri üretilmedi.")

    if all_predictions_test:
        submission = pd.DataFrame({'ImageId': np.arange(1, len(all_predictions_test) + 1), 
                                   'Label': all_predictions_test})
        submission['Label'] = submission['Label'].astype(int)
        submission.to_csv('submission_snn.csv', index=False) # Updated filename
        print("\nTahminler 'submission_snn.csv' dosyasına kaydedildi.")

    if train_accuracies:
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'bo-', label='Eğitim Doğruluğu')
            plt.title('Epoch Bazında Eğitim Doğruluğu') # Updated title
            plt.xlabel('Epoch')
            plt.ylabel('Doğruluk')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Grafik çizilemedi (Grafik arayüzü olmayan bir ortamda çalışıyor olabilirsiniz): {e}")
            
    return snn, train_accuracies, None, all_predictions_test, all_confidences_test


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    snn_model, accuracies, _, test_preds, test_confs = main() # Updated call
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nToplam Çalışma Süresi: {total_time:.2f} saniye")
    if accuracies and len(accuracies) > 0 :
        print(f"Epoch başına ortalama süre: {total_time/len(accuracies):.2f} saniye")
        print(f"Son Eğitim Doğruluğu: {accuracies[-1]:.1%}")
    if snn_model:
        print(f"Son Boltzmann Sıcaklığı: {snn_model.temperature:.4f}")
    if test_confs and len(test_confs) > 0:
        high_conf_preds = np.sum(np.array(test_confs) > 0.8)
        print(f"Yüksek Güvenli Test Tahminleri (>0.8): {high_conf_preds}/{len(test_confs)}")