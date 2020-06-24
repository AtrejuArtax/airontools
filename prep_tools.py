import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import random
import math
import itertools


def to_prep_samples_targets(drivers_file_name, requests_file_name, length, target_delay, start_end_dates,
                            min_max_latitude, min_max_longitude, max_n_samples=None, return_stats=False):
    """From dataframes to a list of preprocessed samples.

        Parameters:
            drivers_file_name (str): Drivers file name.
            requests_file_name (str): Requests file name.
            length (int): Length of the samples.
            target_delay (timedelta): Delay between target request and driver suggestion.
            start_end_dates (list): Start end dates to to filter.
            min_max_longitudes (list): Minimum and maximum longitudes.
            min_max_latitudes (list): Minimum and maximum latitudes.
            start_end_dates (list): Start end dates to to filter.
            max_n_samples (int): Maximum number of samples.
            return_stats (bool): Whether to return stats or not.

        Returns:
            2 pd.DataFrame.
    """

    drivers_features, drivers_date_name = ['lon', 'lat'], 'timestamp'
    requests_features, requests_date_name = ['longitude', 'latitude'], 'created_at'
    reference_lonlat = [sum(min_max_longitude) / 2, sum(min_max_latitude) / 2]

    # Load datasets
    datasets = []
    selected_
    for file_name, feature_info in zip([drivers_file_name, requests_file_name], [drivers_date_name, requests_date_name]):
        def valid(chunks):
            count = 0
            for chunk in chunks:
                if max_n_samples is not None and count >= max_n_samples * length:
                    break
                for key, values in conds.items():
                    if values['type'] == 'date':
                        chunk[key] = pd.to_datetime(chunk[key]).dt.tz_localize(None)
                    chunk = chunk.loc[chunk[key] >= values['cond'][0]]
                    chunk = chunk.loc[chunk[key] <= values['cond'][1]]
                count += chunk.shape[0]
                yield chunk
        conds = {}
        for feature_name, feature_type, min_max in feature_info[0]:
            conds.update({feature_name: {'type': 'date', 'cond': start_end_dates}})
        n_chunks = 10 ** 3
        chunks = pd.read_csv(file_name, chunksize=n_chunks)
        datasets += [pd.concat(valid(chunks=chunks))]
    drivers, requests = datasets

    # Sort input datasets based on time
    drivers[drivers_date_name] = pd.to_datetime(drivers[drivers_date_name])
    drivers[drivers_date_name] = drivers[drivers_date_name].dt.tz_localize(None)
    drivers = drivers.sort_values(drivers_date_name, ascending=True)
    requests[requests_date_name] = pd.to_datetime(requests[requests_date_name])
    requests = requests.sort_values(requests_date_name, ascending=True)

    # Create the samples
    samples, targets, means, stds, requests_features_ = [], [], [], [], []
    for _, driver in drivers.iterrows():

        # Select target based on time delay with respect the driver
        targets_ = requests[requests[requests_date_name] >= driver[drivers_date_name] + target_delay]

        # Continue in case you found a target
        if len(targets_) > 0:

            # Adjust target and select requests based on time delay with respect the driver
            targets_.index = np.arange(0, len(targets_))
            closest, closest_i = get_closest_latlon(
                reference=driver[drivers_features].tolist(),
                candidates=targets_.loc[0:100, requests_features].values.tolist())
            target = pd.Series(
                data=closest + targets_.loc[closest_i, [requests_date_name]].values.tolist(),
                index=targets_.columns)
            requests_ = requests[requests[requests_date_name] <= driver[drivers_date_name]]
            requests_ = requests_.sort_values(requests_date_name, ascending=False)

            # Create new features and pre-process the sample
            if len(requests_) >= length:

                # Adjustments of requests_
                requests_.index = np.arange(0, len(requests_))
                requests_ = requests_.loc[:length - 1, requests_.columns]

                # Make a copy of feature names
                drivers_features_, requests_features_, target_features_ = \
                    drivers_features.copy(), requests_features.copy(), requests_features.copy()

                # From longitude and latitude to x and y as kms
                for new_name, data_info in itertools.product(
                        ['x', 'y'],
                        [['requests_', requests_features_], ['driver', drivers_features_], ['target', target_features_]]):
                    data = requests_ if data_info[0] == 'requests_' else driver if data_info[0] == 'driver' else target
                    reference_lonlat_ = [reference_lonlat[0], None] if new_name == 'x' else [None, reference_lonlat[1]]
                    coord_name = data_info[1][0] if new_name == 'x' else data_info[1][1]
                    new_values = data[coord_name].apply(distance_f, args=(reference_lonlat_,))\
                        if data_info[0] == 'requests_' else distance_f(data[coord_name], reference_lonlat_)
                    data[new_name] = new_values
                    feature_ind = [i for i, feature_name in enumerate(data_info[1]) if feature_name == coord_name][0]
                    data_info[1][feature_ind] = new_name

                # Create new features based on time
                for trigo_type, data_info in itertools.product(
                        ['sinus', 'cosine'], [['requests_', requests_features_], ['driver', drivers_features_]]):
                    data, date_name = [requests_, requests_date_name] \
                        if data_info[0] == 'requests_' else [driver, drivers_date_name]
                    new_values = data[date_name].apply(time_to_trigonometric, args=(trigo_type,))\
                        if data_info[0] == 'requests_' else time_to_trigonometric(data[date_name], trigo_type)
                    data[''.join([date_name, '_', trigo_type, '_teta'])] = new_values
                    data_info[1] += [''.join([date_name, '_', trigo_type, '_teta'])]

                # Mean and standard deviation from the requests since they define where things happen
                mean = requests_.loc[:, requests_features_].mean(axis=0)
                std = requests_.loc[:, requests_features_].std(axis=0)

                # Preprocess (standardize) driver request's data given mean and std
                prep_driver = driver.copy()
                prep_driver[drivers_features_] =\
                    (driver[drivers_features_] - mean.values.tolist()) / std.values.tolist()
                prep_requests = requests_.copy()
                prep_requests.loc[:, requests_features_] =\
                    (requests_.loc[:, requests_features_] - mean) / std
                prep_target = target.copy()
                prep_target[target_features_] = \
                    (prep_target[target_features_] - mean[target_features_]) / std[target_features_]
                samples += [[[prep_driver['id_driver']],
                             prep_driver[drivers_features_].tolist(),
                             prep_requests.loc[:, requests_features_].values.tolist()]]
                targets += [[prep_target[target_features_].tolist()]]
                if return_stats:
                    means += [[mean[requests_features_].tolist()]]
                    stds += [[std[requests_features_].tolist()]]

    # Subsample
    if max_n_samples is not None and max_n_samples < len(targets):
        random_inds = random.sample(list(np.arange(0, len(targets))), max_n_samples)
        random_inds.sort()
        samples = [samples[i] for i in random_inds]
        targets = [targets[i] for i in random_inds]
        if return_stats:
            means = [means[i] for i in random_inds]
            stds = [stds[i] for i in random_inds]

    # Create dataframes
    samples = pd.DataFrame(data=samples, columns=['id_driver', 'driver_features', 'requests_features'])
    targets = pd.DataFrame(data=targets, columns=['driver_target'])
    if return_stats:
        means = pd.DataFrame(data=means, columns=['means'])
        stds = pd.DataFrame(data=stds, columns=['stds'])

    returns = [samples, targets]
    if return_stats:
        returns += [means, stds, requests_features_]
    return returns


def get_closest_latlon(reference, candidates):
    """Get closest candidate considering Earth surface.

        Parameters:
            reference (list): Reference angles [longitude, latitude].
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
            here (list, float): Here point [longitude, latitude].
            there (list): There point [longitude, latitude].

        Returns:
            A list.
    """

    assert(not all([there_ is None for there_ in there]))
    if isinstance(here, float) and any([there_ is None for there_ in there]):
        here, there = [[0, here], [0, there[1]]] if there[0] is None else [[here, 0], [there[0], 0]]

    R = 6373.0
    lon_there, lat_there = math.radians(there[0]), math.radians(there[1])
    lon_here, lat_here = math.radians(here[0]), math.radians(here[1])
    dlon = lon_here - lon_there
    dlat = lat_here - lat_there
    a = math.sin(dlat / 2) ** 2 + math.cos(lat_there) * math.cos(lat_here) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def time_to_trigonometric(date, trigo_type):
    """From time (24 hours clock) to trigonometric value (cosine or sinus).

        Parameters:
            date (datetime): Date.
            trigo_type (str): Trigonometric function to be used.

        Returns:
            A float number.
    """
    radians = math.radians(
        date.hour * float(360) / 24 + date.minute * float(360) / (24 * 60) + date.second * float(
            360) / (24 * 60 * 60))
    if trigo_type == 'sinus':
        return np.sin(radians)
    elif trigo_type == 'cosine':
        return np.cos(radians)
