import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import random
import math


def to_prep_samples_targets(drivers_file_name, requests_file_name, length, target_delay, start_end_dates, n_samples):
    """From dataframes to a list of preprocessed samples.

        Parameters:
            drivers_file_name (str): Drivers file name.
            requests_file_name (str): Requests file name.
            length (int): Length of the samples.
            target_delay (timedelta): Delay between target request and driver suggestion.
            start_end_dates (list): Start end dates to to filter.
            n_samples (int): Number of samples.

        Returns:
            A list.
    """

    # Load datasets
    datasets = []
    for file_name, data_type in zip([drivers_file_name, requests_file_name], ['drivers', 'requests']):
        def valid(chunks):
            for chunk in chunks:
                for key, values in conds.items():
                    if values['type'] == 'date':
                        chunk[key] = pd.to_datetime(chunk[key]).dt.tz_localize(None)
                    chunk = chunk.loc[chunk[key] >= values['cond'][0]]
                    chunk = chunk.loc[chunk[key] <= values['cond'][1]]
                yield chunk
        date_name = 'timestamp' if data_type == 'drivers' else 'created_at'
        conds = {date_name: {'type': 'date', 'cond': start_end_dates}}
        n_chunks = 10 ** 5
        chunks = pd.read_csv(file_name, chunksize=n_chunks)
        datasets += [pd.concat(valid(chunks=chunks))]
    drivers, requests = datasets

    # Sort input datasets based on time
    drivers['timestamp'] = pd.to_datetime(drivers['timestamp'])
    drivers['timestamp'] = drivers['timestamp'].dt.tz_localize(None)
    drivers = drivers.sort_values('timestamp', ascending=True)
    requests['created_at'] = pd.to_datetime(requests['created_at'])
    requests = requests.sort_values('created_at', ascending=True)

    # Create the samples
    samples, targets = [], []
    for _, driver in drivers.iterrows():

        # Select target and requests based on time delay with respect the driver
        target = requests[requests['created_at'] >= driver['timestamp'] + target_delay]
        target.index = np.arange(0, target.shape[0])
        closest, closest_i = get_closest_latlon(
            reference=driver.tolist()[1:3],
            candidates=target.loc[0:100, ['latitude', 'longitude']].values.tolist())
        target = pd.DataFrame(
            data=[closest + target.loc[closest_i, ['created_at']].values.tolist()],
            columns=target.columns)
        requests_ = requests[requests['created_at'] <= driver['timestamp']]
        requests_ = requests_.sort_values('created_at', ascending=False)
        requests_.index = np.arange(0, requests_.shape[0])
        requests_ = requests_.loc[:length, requests_.columns]

        # Create the preprocessed sample
        if requests_.shape[0] >= length:

            # Mean and standard deviation from the requests since they define where things happen
            mean = requests_[:, ['latitude', 'longitude']].mean(axis=0).values.tolist()
            std = requests_[:, ['latitude', 'longitude']].std(axis=0).values.tolist()

            # Preprocess (standardize) driver request's data given mean and std
            prep_driver = driver.copy()
            prep_driver[:, ['latitude', 'longitude']] = (driver[:, ['lat', 'lon']] - mean) / std
            prep_requests = requests_.copy()
            prep_requests[:, ['latitude', 'longitude']] = (requests_[:, ['latitude', 'longitude']] - mean) / std
            prep_target = target.copy()
            prep_requests[:, ['latitude', 'longitude']] = (target[:, ['latitude', 'longitude']] - mean) / std
            samples += [prep_driver.tolist(), prep_requests.values.tolist()]
            targets += prep_target.tolist()

    if n_samples < len(targets):
        random_inds = random.sample(list(np.arange(0, len(targets))), n_samples)
        samples = [samples[i] for i in random_inds]
        targets = [targets[i] for i in random_inds]

    return samples, targets


def get_closest_latlon(reference, candidates):
    """Get closest candidate considering Earth surface.

        Parameters:
            reference (list): Reference angles.
            candidates (list): Candidates.

        Returns:
            A list and an integer.
    """
    closest, closest_i = 2 * [None]
    closest_distance = np.inf
    for candidate, i in zip(candidates, np.arange(0, len(candidates))):
        distance = distance_f(here=reference, there=candidate)
        if distance < closest_distance:
            closest_distance = distance
            closest = candidate
            closest_i = i
    return closest, closest_i


def distance_f(here, there):
    """Distance between two points on the Earth's surface.

        Parameters:
            here (list): Here point.
            there (list): There point.

        Returns:
            A list.
    """
    R = 6373.0
    lat1 = math.radians(there[0])
    lon1 = math.radians(there[1])
    lat2 = math.radians(here[0])
    lon2 = math.radians(here[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
