import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

from typing import List

def _compute_cross_band_importance(bands : List[List[float]], model : torch.nn.Module, dataloader : DataLoader, model_device : torch.device, sampling_rate: int = 100):    

    for i in range(len(bands)):
        assert len(bands[i]) == 2

    y_pred = []
    y_true = []
    band_importance = []
    time_importance = []

    for batch in dataloader:
        inputs, y_true_batch = batch
        
        # store the true label of the input element
        y_true.append(y_true_batch.numpy())

        # compute the prediction of the model
        pred_proba = F.softmax(model(inputs.to(model_device)).cpu()).detach().numpy()       
        y_pred.append( np.argmax( pred_proba, axis = -1) )
        n_class = pred_proba.shape[-1]

        # port the input to numpy
        inputs = inputs.cpu().detach().numpy()
        batch_size, seq_len, n_channels, n_samples = inputs.shape

        # reshape the input to consider only the input signal
        filtered_inputs = inputs.copy()
        filtered_inputs = filtered_inputs.reshape(-1, seq_len * n_samples)


        # remove the frequency band from the input using scipy       
        for band in bands:
            # filter bandstop - reject the frequencies specified in freq_band
            lowcut = band[0]
            highcut = band[1]
            order = 4
            nyq = 0.5 * sampling_rate
            low = lowcut / nyq
            high = highcut / nyq
            sos = signal.butter(order, [low, high], btype='bandstop', output='sos')

            for index in range(batch_size):     
                filtered_inputs[index] = signal.sosfilt(sos, filtered_inputs[index])

        # reshape the input signal to the original size and port it to tensor
        filtered_inputs = filtered_inputs.reshape(batch_size, seq_len, n_channels, n_samples)
        filtered_inputs = torch.from_numpy(filtered_inputs)
        inputs = torch.from_numpy(inputs)

        # compute the prediction of the model with the filtered input, the prediction is a tensor of size batch_size * seq_len, n_classes
        batch_importance = F.softmax(model(filtered_inputs.to(model_device)).cpu()).detach().numpy()

        # the importance is the difference between the prediction with the original input and the prediction with the filtered input
        batch_importance = pred_proba - batch_importance
        band_importance.append(batch_importance)

        ig = IntegratedGradients(model)

        partial_time_importance = []
        for c in range(n_class):
            partial_time_importance.append(ig.attribute(inputs.to(model_device), filtered_inputs.to(model_device), target=c).cpu().numpy())
        
        time_importance.append(partial_time_importance)

    # reshape the lists to ignore the batch_size dimension
    y_pred = np.concatenate(y_pred).reshape(-1)
    y_true = np.concatenate(y_true).reshape(-1)
    band_importance = np.concatenate(band_importance).reshape(-1, n_class)

    return time_importance, band_importance, y_pred, y_true

#RICORDA DI LEVARE I PRIMI DUE PARAMETRI
def compute_band_importance(bands : List[List[float]],  
                            model : torch.nn.Module, dataloader : DataLoader , model_device : torch.device, sampling_rate: int = 100, class_names : List[str] = ["Wake", "N1", "N2", "DS", "REM"], average_type : int = 0):
    
    for i in range(len(bands)):
        assert len(bands[i]) == 2
    assert len(band_names) == 6
    assert len(class_names) == 5
    assert average_type == 0 or average_type == 1

    # the dataloader is recreated here with the parameter shuffle = False in order to have consistency with the order of data
    # this allows us to be precise in calculating different importances referring to the same samples

    dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
    )

    band_freq_combinations = []
    band_combinations_dict = {}
    band_time_combinations_dict = {}
    permutations_array = []

    for i in range(len(bands)):
        combination_list = it.combinations(bands, i+1)
        for elem in combination_list:
            band_freq_combinations.append(elem)
    
    for cross_band in band_freq_combinations:
        permuted_bands = np.zeros( len( bands ) )

        for i, band in enumerate( bands ):
            if band in cross_band:
                permuted_bands [i] = 1
        
        print(permuted_bands)
        permutations_array.append(permuted_bands)
        time_importance, band_importance, y_pred, y_true = _compute_cross_band_importance(cross_band, model, dataloader, model_device, sampling_rate)

        band_combinations_dict[str(permuted_bands)] = band_importance
        band_time_combinations_dict[str(permuted_bands)] = time_importance

    permuted_bands_importance = []
    permuted_bands_time_importance = []

    for i in range(len(permutations_array)):
        key = permutations_array[i]
        permuted_bands_importance.append(band_combinations_dict[str(key)])
        permuted_bands_time_importance.append(band_time_combinations_dict[str(key)])

    importances_matrix = []
    time_importances_matrix = []

    for i in range(len(bands)):

        #simple_average
        if average_type == 0:
            band_importance = get_simple_importance(permuted_bands_importance, permutations_array, i)
            band_time_importance = get_simple_importance(permuted_bands_time_importance, permutations_array, i)
        #weighted_average
        elif average_type == 1:
            band_importance = get_weighted_importance(permuted_bands_importance, permutations_array, i)
            band_time_importance = get_weighted_importance(permuted_bands_time_importance, permutations_array, i)

        importances_matrix.append(band_importance)
        time_importances_matrix.append(band_time_importance)

    return time_importances_matrix, importances_matrix, y_pred, y_true
