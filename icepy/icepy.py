import regionmask
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

OCEAN_AREA = 3.618 * 10**14
ICE_DENSITY = 916.75
OCEAN_DENSITY = 1027
EARTH_RADIUS = 6.371 * 10**6


def calc_sphere_cap_proportion(polar_lat):
    assert np.all(polar_lat >= 0) and np.all(
        polar_lat <= 180
    ), "polar latitude must be in range [0,180] degrees"
    return (1 - np.cos(polar_lat * np.pi / 180)) / 2


def calc_lat_midpoints(lats):
    assert np.all(np.diff(lats) > 0), "latitudes must be monotonically increasing"
    assert np.all(lats >= -90) and np.all(
        lats <= 90
    ), "latitudes must be in range (-90,90) degrees"
    # assert lats[0] > 0, "areas cell at 0 latitude is undefined"
    # assert lats[-1] < 180, "areas cell at 180 latitude is undefined"
    midpoints = np.zeros(len(lats) + 1)
    midpoints[1:-1] = lats[:-1] + ((lats[1:] - lats[:-1]) / 2)
    midpoints[0] = -90
    midpoints[-1] = 90

    return midpoints


def calc_lon_midpoints(lons):
    assert np.all(np.diff(lons) > 0), "longitudes must be monotonically increasing"
    assert np.all(lons >= -180) and np.all(
        lons <= 180
    ), "longitudes must be in range (-180,180) degrees"
    midpoints = np.zeros(len(lons))
    midpoints[:-1] = lons[:-1] + ((lons[1:] - lons[:-1]) / 2)
    midpoints[-1] = (lons[-1] + ((lons[0] - lons[-1] + 360) / 2)) % 360
    midpoints = np.sort(midpoints)

    return midpoints


def calc_grid_area_geo(lons, lats, earth_area=None):
    """
    A matrix of the area associated with each thickness datapoint
    """
    if earth_area is None:
        earth_area = 4 * np.pi * EARTH_RADIUS**2

    # calculate coordinate middpoints, since thickness represents value at centre of an area
    lat_midpoints = calc_lat_midpoints(lats)
    cap_proportions = calc_sphere_cap_proportion(lat_midpoints - lat_midpoints[0])
    strip_proportions = cap_proportions[1:] - cap_proportions[:-1]

    lon_midpoints = calc_lon_midpoints(lons)

    lat_proportions = np.zeros(len(lon_midpoints))
    lat_proportions[:-1] = (lon_midpoints[1:] - lon_midpoints[:-1]) / 360
    lat_proportions[-1] = (lon_midpoints[0] - (lon_midpoints[-1] - 360)) / 360
    cell_proportions = np.matmul(
        strip_proportions.reshape(-1, 1), lat_proportions.reshape(1, -1)
    )

    np.testing.assert_almost_equal(np.sum(cell_proportions), 1)
    cell_areas = earth_area * cell_proportions

    return cell_areas


def volume_to_sle(volume):
    sle = (volume * ICE_DENSITY) / (
        OCEAN_AREA * OCEAN_DENSITY
    )
    return sle


def sle_to_volume(sle):
    volume = (
        sle * OCEAN_AREA * OCEAN_DENSITY / ICE_DENSITY
    )
    return volume

def region_mask(dataarray, region):
    Eurasia = [
        [-20, 30],
        [-20, 60],
        [-10, 60],
        [-10, 65],
        [0, 80],
        [0, 90],
        [170, 90],
        [170, 30],
    ]
    NorthAmerica = [
        [-47, 30],
        [-47, 50],
        [-60, 70],
        [-70, 75],
        [-70, 90],
        [-170, 90],
        [-170, 30],
    ]
    Greenland = [
        [-47, 50],
        [-60, 70],
        [-70, 75],
        [-70, 90],
        [0, 90],
        [0, 80],
        [-10, 65],
        [-10, 60],
        [-20, 60],
    ]
    Antarctica = [[-180, -90], [-180, -60], [180, -60], [180, -90]]
    Other = [
        [-180, -60],
        [-180, 90],
        [-170, 90],
        [-170, 30],
        [-47, 30],
        [-47, 50],
        [-20, 60],
        [-20, 30],
        [170, 30],
        [170, 90],
        [180, 90],
        [180, -60],
        [-180, -60],
    ]

    names = ["Eurasia", "North America", "Greenland", "Antarctica", "Other"]
    abbrevs = ["Er", "NA", "Gr", "An", "O"]

    mask_generator = regionmask.Regions(
        [Eurasia, NorthAmerica, Greenland, Antarctica, Other],
        names=names,
        abbrevs=abbrevs,
        name="Ice Sheet Regions",
    )
    mask = mask_generator.mask(dataarray)

    if region in names:
        index = names.index(region)
    elif region in abbrevs:
        index = names.index(region)
    elif isinstance(region, int):
        index = region
    else:
        raise ValueError(f"Invalid region: {region}")

    mask = xr.ones_like(mask).where(mask == index, 0).values

    return mask


def single_stretch(x, new_min, new_max):
    x_norm = (x - x.min()) / (x.max() - x.min())
    x_new = x_norm * (new_max - new_min) + new_min
    return x_new


def double_stretch(x, centre, new_centre, new_min, new_max):

    new_centre_id = np.abs(x - new_centre).argmin()
    x_left = single_stretch(x[x <= centre], new_min=new_min, new_max=new_centre)
    x_right = single_stretch(
        x[x > centre], new_min=x[new_centre_id + 1], new_max=new_max
    )
    x_new = np.concatenate([x_left, x_right])
    return x_new


def stretch(
    x,
    y,
    centre,
    new_centre,
    left_tie=None,
    right_tie=None,
    new_left_tie=None,
    new_right_tie=None,
):

    if left_tie is None:
        left_tie = x[0]
    if right_tie is None:
        right_tie = x[-1]
    if new_left_tie is None:
        new_left_tie = left_tie
    if new_right_tie is None:
        new_right_tie = right_tie

    # check for monotonic decrease
    if (np.diff(x) <= 0).all():
        left_tie, right_tie = right_tie, left_tie
        new_left_tie, new_right_tie = new_right_tie, new_left_tie

    # interpolate original data onto important points
    original_interp = interp1d(x=x, y=y)
    x_mod = np.unique(
        np.insert(
            x, 0, [centre, new_centre, left_tie, right_tie, new_left_tie, new_right_tie]
        )
    )
    y_mod = original_interp(x_mod)

    # select points in tie range and stretch
    cond = (x_mod >= left_tie) & (x_mod <= right_tie)
    x_mod[cond] = double_stretch(
        x=x_mod[cond],
        centre=centre,
        new_centre=new_centre,
        new_min=left_tie,
        new_max=right_tie,
    )

    interp = interp1d(x=x_mod, y=y_mod)
    x_mod = np.unique(
        np.insert(x_mod, 0, [left_tie, right_tie, new_left_tie, new_right_tie])
    )
    y_mod = interp(x_mod)

    # modify based on new tie points
    if new_right_tie > right_tie:
        cond = (x_mod >= new_centre) & (x_mod <= right_tie)
        remove_cond = (x_mod >= right_tie) & (x_mod <= new_right_tie)
        x_mod[cond] = single_stretch(
            x=x_mod[cond], new_min=new_centre, new_max=new_right_tie
        )
        x_mod = np.delete(x_mod, remove_cond)
        y_mod = np.delete(y_mod, remove_cond)
    elif new_right_tie < right_tie:
        cond = (x_mod >= new_centre) & (x_mod <= right_tie)
        x_mod[cond] = single_stretch(
            x=x_mod[cond], new_min=new_centre, new_max=new_right_tie
        )

    if new_left_tie > left_tie:
        cond = (x_mod <= new_centre) & (x_mod >= left_tie)
        x_mod[cond] = single_stretch(
            x=x_mod[cond], new_min=new_left_tie, new_max=new_centre
        )

    elif new_left_tie < left_tie:
        cond = (x_mod <= new_centre) & (x_mod >= left_tie)
        remove_cond = (x_mod < left_tie) & (x_mod >= new_left_tie)
        x_mod[cond] = single_stretch(
            x=x_mod[cond], new_min=new_left_tie, new_max=new_centre
        )
        x_mod = np.delete(x_mod, remove_cond)
        y_mod = np.delete(y_mod, remove_cond)

    # interpolate onto original coords
    y_new = interp1d(x=x_mod, y=y_mod)(x)

    return y_new


def gen_time_to_global_vol(times, sea_level, modern_ice_vol, beyond_modern_melt=False):
    if 0 not in times:
        times, sea_level = np.insert(times, 0, 0), np.insert(sea_level, 0, 0)
    ice_vol = modern_ice_vol - sea_level

    if beyond_modern_melt is False:
        ice_vol[ice_vol < modern_ice_vol] = modern_ice_vol

    get_global_budget = interp1d(x=times, y=ice_vol)
    return get_global_budget


def get_masks(da):
    regions = ["Eurasia", "North America", "Greenland", "Antarctica"]
    masks = {region: region_mask(da, region) for region in regions}

    total_mask = np.asarray([mask for mask in masks.values()]).sum(axis=0)
    total_mask[total_mask > 0] = 1
    remainder_mask = (~total_mask.astype(bool)).astype(int)

    masks["Other"] = remainder_mask
    masks["Global"] = np.ones_like(masks["Eurasia"])

    return masks


def get_regional_vols(ice_da, as_da=True):
    grid_area = calc_grid_area_geo(ice_da.lon.values, ice_da.lat.values)
    masks = get_masks(da=ice_da)

    regional_vols = {
        region: volume_to_sle((ice_da * grid_area * mask).sum(dim=["lon", "lat"]))
        for region, mask in masks.items()
    }

    if as_da is False:
        regional_vols = {key: values.values for key, values in regional_vols.items()}

    return regional_vols


def get_regional_slices(ice_da, as_da=True):
    masks = get_masks(da=ice_da)

    regional_slices = {region: ice_da * mask for region, mask in masks.items()}
    if as_da is False:
        regional_slices = {
            key: values.values for key, values in regional_slices.items()
        }

    return regional_slices


def extract_deglaciation(ice_da):
    grid_area = calc_grid_area_geo(ice_da.lon.values, ice_da.lat.values)
    global_vols = (ice_da * grid_area).sum(dim=["lon", "lat"])

    max_time = global_vols.time[global_vols.argmax()]

    return ice_da.sel(time=slice(max_time, 0))


class multi_interp1d:
    def __init__(self, x, y, keys):
        self.interps = {
            key: interp1d(x=x, y=y[idx], fill_value="extrapolate")
            for idx, key in enumerate(keys)
        }

    def __call__(self, x_new):
        y_new = {key: interp(x_new) for key, interp in self.interps.items()}
        return y_new


class parallel_interp1d:
    def __init__(self, x, y, keys):
        self.interps = {
            key: interp1d(x=x[idx], y=y[idx], fill_value="extrapolate", axis=0)
            for idx, key in enumerate(keys)
        }

    def __call__(self, x_new, combine=False, exclude=None):
        y_new = {key: self.interps[key](values) for key, values in x_new.items()}
        if exclude is not None:
            y_new.pop(exclude)
        if combine is True:
            combined = np.stack(list(y_new.values())).sum(axis=0)
            return combined
        else:
            return y_new

        idx_sorted = np.argsort(volumes)
        volumes_sorted = volumes[idx_sorted]
        ice_sorted = ice[idx_sorted]

        if volumes_sorted[0] != 0:
            volumes_sorted = np.insert(volumes_sorted, 0, 0, axis=0)
            ice_sorted = np.insert(ice_sorted, 0, np.zeros_like(ice_sorted[0]), axis=0)


def gen_global_vol_to_regional_vol(global_vol, regional_vols):
    x = np.insert(global_vol, 0, 0)
    y = [np.insert(i, 0, 0) for i in list(regional_vols.values())]
    global_vol_to_regional_vol = multi_interp1d(
        x=x, y=y, keys=list(regional_vols.keys())
    )

    return global_vol_to_regional_vol


def gen_time_to_model_slice(ice_da):
    time_to_model_slice = interp1d(x=ice_da.time.values, y=ice_da.values, axis=0)

    return time_to_model_slice


def gen_regional_vol_to_model_slice(ice_da):
    regional_vols = get_regional_vols(ice_da, as_da=False)
    regional_slices = get_regional_slices(ice_da, as_da=False)

    for region in regional_vols.keys():
        # sort all the slices
        regional_vols[region], idx_unique = np.unique(
            regional_vols[region], return_index=True
        )
        regional_slices[region] = regional_slices[region][idx_unique]

        # check for a zero-volume slice, add if not
        if regional_vols[region][0] != 0:
            regional_vols[region] = np.insert(regional_vols[region], 0, 0, axis=0)
            regional_slices[region] = np.insert(
                regional_slices[region],
                0,
                np.zeros_like(regional_slices[region][0]),
                axis=0,
            )

    regional_vol_to_model_slice = parallel_interp1d(
        x=list(regional_vols.values()),
        y=list(regional_slices.values()),
        keys=list(regional_vols.keys()),
    )

    return regional_vol_to_model_slice


def get_times():
    pass


def insert_slices(x, y, x_insert, y_insert, mask=None):
    if mask is None:
        mask = np.ones_like(y[0])

    idx_array = [np.where(np.isclose(x, v)) for v in x_insert]
    idx_array = [v[0][0] for v in idx_array if v[0].size > 0]
    assert len(idx_array) == len(x_insert)

    y_subset = y[idx_array]
    y_subset[:, mask.astype(bool)] = y_insert[:, mask.astype(bool)]
    y[idx_array] = y_subset

    return y


def get_history_volume(ice_history, lons, lats):
    return volume_to_sle(
        np.sum(
            ice_history * calc_grid_area_geo(lons=lons, lats=lats),
            axis=(1, 2),
        )
    )


def time_generator(periods):
    times = []
    period_times = {}
    for period, (t_bounds, interval) in periods.items():
        t_range = t_bounds[0] - t_bounds[1]
        period_time = -np.arange(-t_bounds[0], -(t_bounds[1] - interval), interval)
        period_time = period_time[
            (period_time <= t_bounds[0]) & (period_time >= t_bounds[1])
        ]
        period_times[period] = period_time
        times.append(period_time)
    times = np.concatenate(times)
    times = np.flip(np.unique(np.round(times, 3)))
    return times, period_times


def filter_da(ice_da, modern_ice_vol):
    cond = (
        get_history_volume(
            ice_da.values, lons=ice_da.lon.values, lats=ice_da.lat.values
        )
        >= modern_ice_vol
    )
    return ice_da[cond]


def generate_curve(
    naer_budget,
    times,
    er_pgm_t,
    er_lig_t,
    na_pgm_t,
    na_lig_t,
    er_pgm_vol,
    discrep=False,
):
    stretch_centre = 137
    stretch_left_tie = 145
    stretch_right_tie = 128
    max_duration = 2
    er_ratio = er_pgm_vol / naer_budget.max()

    idx_subset = (times <= 220) & (times >= 122)
    times_subset = times[idx_subset]

    # north america custom insert
    pgp_na = stretch(
        x=times,
        y=naer_budget * (1 - er_ratio),
        centre=stretch_centre,
        new_centre=na_pgm_t,
        left_tie=stretch_left_tie,
        right_tie=stretch_right_tie,
        new_right_tie=na_lig_t,
    )

    pgp_na[times >= na_pgm_t + max_duration] = stretch(
        x=times,
        y=naer_budget * (1 - er_ratio),
        centre=stretch_centre,
        new_centre=na_pgm_t + max_duration,
        left_tie=stretch_left_tie,
        right_tie=stretch_right_tie,
        new_right_tie=na_lig_t,
    )[times >= na_pgm_t + max_duration]

    pgp_na[(times <= na_pgm_t + max_duration) & (times >= na_pgm_t)] = (
        naer_budget * (1 - er_ratio)
    ).max()

    # eurasia custom insert
    pgp_er = stretch(
        x=times,
        y=naer_budget * er_ratio,
        centre=stretch_centre,
        new_centre=er_pgm_t,
        left_tie=stretch_left_tie,
        right_tie=stretch_right_tie,
        new_right_tie=er_lig_t,
    )

    pgp_er[times >= er_pgm_t + max_duration] = stretch(
        x=times,
        y=naer_budget * er_ratio,
        centre=stretch_centre,
        new_centre=er_pgm_t + max_duration,
        left_tie=stretch_left_tie,
        right_tie=stretch_right_tie,
        new_right_tie=er_lig_t,
    )[times >= er_pgm_t + max_duration]

    pgp_er[(times <= er_pgm_t + max_duration) & (times >= er_pgm_t)] = (
        naer_budget * (er_ratio)
    ).max()
    if discrep is True:
        budget_discrep = naer_budget[idx_subset] - (pgp_na + pgp_er)[idx_subset]
        return pgp_na, pgp_er, budget_discrep, times_subset
    else:
        return pgp_na, pgp_er


def validate_curve(
    naer_budget,
    times,
    sea_level_btm_err_interp,
    sea_level_top_err_interp,
    er_pgm_t,
    er_lig_t,
    na_pgm_t,
    na_lig_t,
    er_pgm_vol,
):
    pgp_na, pgp_er, budget_discrep, times_subset = generate_curve(
        naer_budget,
        times,
        er_pgm_t,
        er_lig_t,
        na_pgm_t,
        na_lig_t,
        er_pgm_vol,
        discrep=True,
    )
    valid = (sea_level_btm_err_interp(times_subset) < budget_discrep).all() & (
        sea_level_top_err_interp(times_subset) > budget_discrep
    ).all()
    return valid
