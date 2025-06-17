# Author: Trung Le
# Original file available at https://github.com/trungle93/STNDT
# Adapted by Hau Pham
"""
JAX/Equinox implementation of dataset utilities for STNDT
Translated from PyTorch implementation
"""

import os
import os.path as osp
import h5py
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, List
import logging
import jax


class DATASET_MODES:
    train = "train"
    val = "val" 
    test = "test"
    trainval = "trainval"


def merge_train_valid(train_data, valid_data, train_ixs, valid_ixs):
    """
    Merge training and validation data using indices
    Translated from the original utility
    """
    if train_data.shape[0] == train_ixs.shape[0] and valid_data.shape[0] == valid_ixs.shape[0]:
        # If indices match up, use them to merge
        data = np.full_like(np.concatenate([train_data, valid_data]), np.nan)
        if min(min(train_ixs), min(valid_ixs)) > 0:
            # MATLAB data (1-indexed)
            train_ixs -= 1
            valid_ixs -= 1
        data[train_ixs.astype(int)] = train_data
        data[valid_ixs.astype(int)] = valid_data
    else:
        # If indices don't match, check if data is the same
        if np.all(train_data == valid_data):
            data = train_data
        else:
            raise ValueError(f"Shape mismatch: Index shape {train_ixs.shape} "
                           f"does not match data shape {train_data.shape}")
    return data


class JAXSpikesDataset:
    """
    JAX implementation of SpikesDataset
    Loads neural spike data and converts to JAX arrays
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        filename: str,
        mode: str = DATASET_MODES.train,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger
        if self.logger:
            self.logger.info(f"Loading {filename} in {mode}")
        
        self.config = config.get('DATA', {})
        self.model_config = config.get('MODEL', {})
        self.train_config = config.get('TRAIN', {})
        
        self.use_lograte = self.model_config.get('LOGRATE', False)
        self.batch_size = self.train_config.get('BATCH_SIZE', 64)
        
        # Construct file path
        datapath = self.config.get('DATAPATH', 'data/')
        self.datapath = osp.join(datapath, filename)
        
        # Check file extension
        split_path = self.datapath.split(".")
        self.has_rates = False
        self.has_heldout = False
        
        if len(split_path) == 1 or split_path[-1] == "h5":
            spikes, rates, heldout_spikes, forward_spikes = self._get_data_from_h5(mode)
            
            # Convert to JAX arrays
            spikes = jnp.array(spikes, dtype=jnp.int32)
            if rates is not None:
                rates = jnp.array(rates, dtype=jnp.float32)
            if heldout_spikes is not None:
                self.has_heldout = True
                heldout_spikes = jnp.array(heldout_spikes, dtype=jnp.int32)
                forward_spikes = jnp.array(forward_spikes, dtype=jnp.int32)
                
        elif split_path[-1] == "pth":
            # For PyTorch files, we'd need to implement loading
            # For now, raise an error since we're focusing on h5
            raise NotImplementedError("JAX implementation doesn't support .pth files yet")
        else:
            raise ValueError(f"Unknown dataset extension {split_path[-1]}")
        
        # Store dataset properties
        self.num_trials, _, self.num_neurons = spikes.shape
        self.full_length = self.model_config.get('TRIAL_LENGTH', -1) <= 0
        self.trial_length = spikes.shape[1] if self.full_length else self.model_config.get('TRIAL_LENGTH', 100)
        
        if self.has_heldout:
            self.num_neurons += heldout_spikes.shape[-1]
            self.trial_length += forward_spikes.shape[1]
        
        # Batchify data
        self.spikes = self._batchify(spikes)
        self.rates = self._batchify(rates) if self.has_rates else jnp.zeros_like(self.spikes, dtype=jnp.float32)
        self.heldout_spikes = self._batchify(heldout_spikes) if self.has_heldout else jnp.zeros_like(self.spikes)
        self.forward_spikes = self._batchify(forward_spikes) if self.has_heldout else jnp.zeros_like(self.spikes)
        
        # Handle data subsets
        self._apply_data_subsets(mode)
        
    def _get_data_from_h5(self, mode: str) -> Tuple[np.ndarray, ...]:
        """Load data from HDF5 file"""
        with h5py.File(self.datapath, 'r') as h5file:
            h5dict = {key: h5file[key][()] for key in h5file.keys()}
            
            if 'eval_spikes_heldin' in h5dict:  # NLB data format
                get_key = lambda key: h5dict[key].astype(np.float32)
                
                train_data = get_key('train_spikes_heldin')
                train_data_fp = get_key('train_spikes_heldin_forward')
                train_data_heldout_fp = get_key('train_spikes_heldout_forward')
                train_data_all_fp = np.concatenate([train_data_fp, train_data_heldout_fp], -1)
                
                valid_data = get_key('eval_spikes_heldin')
                train_data_heldout = get_key('train_spikes_heldout')
                
                if 'eval_spikes_heldout' in h5dict:
                    valid_data_heldout = get_key('eval_spikes_heldout')
                else:
                    valid_data_heldout = np.zeros(
                        (valid_data.shape[0], valid_data.shape[1], train_data_heldout.shape[2]),
                        dtype=np.float32
                    )
                
                if 'eval_spikes_heldin_forward' in h5dict:
                    valid_data_fp = get_key('eval_spikes_heldin_forward')
                    valid_data_heldout_fp = get_key('eval_spikes_heldout_forward')
                    valid_data_all_fp = np.concatenate([valid_data_fp, valid_data_heldout_fp], -1)
                else:
                    valid_data_all_fp = np.zeros(
                        (valid_data.shape[0], train_data_fp.shape[1], 
                         valid_data.shape[2] + valid_data_heldout.shape[2]),
                        dtype=np.float32
                    )
                
                # NLB data doesn't have ground truth rates
                if mode == DATASET_MODES.train:
                    return train_data, None, train_data_heldout, train_data_all_fp
                elif mode == DATASET_MODES.val:
                    return valid_data, None, valid_data_heldout, valid_data_all_fp
                    
            # Original LFADS-type datasets
            train_data = h5dict['train_data'].astype(np.float32).squeeze()
            valid_data = h5dict['valid_data'].astype(np.float32).squeeze()
            train_rates = None
            valid_rates = None
            
            if "train_truth" in h5dict and "valid_truth" in h5dict:
                self.has_rates = True
                train_rates = h5dict['train_truth'].astype(np.float32)
                valid_rates = h5dict['valid_truth'].astype(np.float32)
                train_rates = train_rates / h5dict['conversion_factor']
                valid_rates = valid_rates / h5dict['conversion_factor']
                
                if self.use_lograte:
                    log_epsilon = self.config.get('LOG_EPSILON', 1e-7)
                    train_rates = np.log(train_rates + log_epsilon)
                    valid_rates = np.log(valid_rates + log_epsilon)
                    
            if mode == DATASET_MODES.train:
                return train_data, train_rates, None, None
            elif mode == DATASET_MODES.val:
                return valid_data, valid_rates, None, None
            elif mode == DATASET_MODES.trainval:
                # Merge training and validation data
                if 'train_inds' in h5dict and 'valid_inds' in h5dict:
                    train_inds = h5dict['train_inds'].squeeze()
                    valid_inds = h5dict['valid_inds'].squeeze()
                    file_data = merge_train_valid(train_data, valid_data, train_inds, valid_inds)
                    if self.has_rates:
                        merged_rates = merge_train_valid(train_rates, valid_rates, train_inds, valid_inds)
                else:
                    if self.logger:
                        self.logger.info("No indices found for merge. Concatenating training and validation samples.")
                    file_data = np.concatenate([train_data, valid_data], axis=0)
                    if self.has_rates:
                        merged_rates = np.concatenate([train_rates, valid_rates], axis=0)
                        
                return file_data, merged_rates if self.has_rates else None, None, None
            else:  # test unsupported
                return None, None, None, None
    
    def _batchify(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Chop data into uniform sizes as configured by trial_length
        
        Args:
            x: Input data array
            
        Returns:
            Reshaped array as (num_samples, trial_length, neurons)
        """
        if x is None or self.full_length:
            return x
            
        trial_time = x.shape[1]
        samples_per_trial = trial_time // self.trial_length
        
        if trial_time % self.trial_length != 0:
            if self.logger:
                self.logger.debug(f"Trimming dangling trial info. Data trial length {trial_time} "
                                f"is not divisible by asked length {self.trial_length}")
        
        # Trim to exact multiple of trial_length
        trimmed_length = samples_per_trial * self.trial_length
        x = x[:, :trimmed_length]
        
        # Reshape to (num_samples, trial_length, neurons)
        reshaped = x.reshape(x.shape[0], samples_per_trial, self.trial_length, x.shape[2])
        return reshaped.reshape(-1, self.trial_length, x.shape[2])
    
    def _apply_data_subsets(self, mode: str):
        """Apply data subsetting for testing or overfitting"""
        if self.config.get('OVERFIT_TEST', False):
            if self.logger:
                self.logger.warning("Overfitting mode: using only 2 samples")
            self.spikes = self.spikes[:2]
            self.rates = self.rates[:2]
            self.num_trials = 2
            
        elif (hasattr(self.config, 'RANDOM_SUBSET_TRIALS') and 
              self.config.get('RANDOM_SUBSET_TRIALS', 1.0) < 1.0 and 
              mode == DATASET_MODES.train):
            
            subset_ratio = self.config['RANDOM_SUBSET_TRIALS']
            if self.logger:
                self.logger.warning(f"Training on {subset_ratio} of the data")
            
            reduced = int(self.num_trials * subset_ratio)
            # For reproducibility, use a deterministic permutation
            seed = self.config.get('SEED', 42)
            key = jax.random.PRNGKey(seed)
            random_subset = jax.random.choice(key, self.num_trials, (reduced,), replace=False)
            
            self.num_trials = reduced
            self.spikes = self.spikes[random_subset]
            self.rates = self.rates[random_subset]
    
    def get_num_neurons(self) -> int:
        return self.num_neurons
    
    def __len__(self) -> int:
        return self.spikes.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[jnp.ndarray, ...]:
        """Return spikes and rates for given index"""
        return (
            self.spikes[index],
            None if self.rates is None else self.rates[index],
            None if self.heldout_spikes is None else self.heldout_spikes[index],
            None if self.forward_spikes is None else self.forward_spikes[index]
        )
    
    def get_dataset(self) -> Tuple[jnp.ndarray, ...]:
        """Return full dataset arrays"""
        return self.spikes, self.rates, self.heldout_spikes, self.forward_spikes
    
    def get_max_spikes(self) -> int:
        """Get maximum spike count in dataset"""
        return int(jnp.max(self.spikes))
    
    def get_num_batches(self) -> int:
        """Get number of batches for given batch size"""
        return self.spikes.shape[0] // self.batch_size
    
    def clip_spikes(self, max_val: int):
        """Clip spike counts to maximum value"""
        self.spikes = jnp.clip(self.spikes, max=max_val)
    
    def get_batches(self, batch_size: Optional[int] = None) -> List[Tuple[jnp.ndarray, ...]]:
        """
        Get data in batches for training
        
        Args:
            batch_size: Batch size (uses config default if None)
            
        Returns:
            List of batches
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        num_samples = len(self)
        batches = []
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_spikes = self.spikes[i:end_idx]
            batch_rates = self.rates[i:end_idx] if self.has_rates else None
            batch_heldout = self.heldout_spikes[i:end_idx] if self.has_heldout else None
            batch_forward = self.forward_spikes[i:end_idx] if self.has_heldout else None
            
            batches.append((batch_spikes, batch_rates, batch_heldout, batch_forward))
        
        return batches


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration for JAX datasets"""
    return {
        'DATA': {
            'DATAPATH': 'data/',
            'LOG_EPSILON': 1e-7,
            'OVERFIT_TEST': False,
            'RANDOM_SUBSET_TRIALS': 1.0,
            'SEED': 42
        },
        'MODEL': {
            'TRIAL_LENGTH': 100,
            'LOGRATE': False
        },
        'TRAIN': {
            'BATCH_SIZE': 64
        }
    }