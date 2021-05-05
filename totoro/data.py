import pathlib
import astropy.table as at
import astropy.units as u
from matplotlib.path import Path
import numpy as np
from pyia import GaiaData

from .config import (apogee_parent_filename, galah_parent_filename,
                     cache_path, plot_path)


class Dataset:

    _id_column = None
    _radial_velocity_name = None
    _elem_err_fmt = None

    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, '_radial_velocity_name'):
            cls._radial_velocity_name = 'radial_velocity'

        for name in ['_id_column', '_elem_err_fmt']:
            if getattr(cls, name) is None:
                raise ValueError(f'You must specify class param: {name}')

        super().__init_subclass__(**kwargs)

    def __init__(self, filename_or_tbl):
        if (isinstance(filename_or_tbl, str)
                or isinstance(filename_or_tbl, pathlib.Path)):
            self.t = at.QTable.read(filename_or_tbl)
        else:
            self.t = at.QTable(filename_or_tbl)
        self.t = self._init_mask()

        # Abundance ratios should be all caps:
        for col in self.t.colnames:
            if ((col.upper().endswith('_FE') or
                    col.upper().startswith('FE_') or
                    col.upper().endswith('_H')) and
                    not col.upper().startswith('FLAG')):
                self.t.rename_column(col, col.upper())

        # Abundance error columns should be _ERR like APOGEE:
        for elem in self.elem_ratios:
            col1 = self._elem_err_fmt.format(elem_name=elem)
            col2 = APOGEEDataset._elem_err_fmt.format(elem_name=elem)
            if col1 in self.t.colnames:
                self.t.rename_column(col1, col2)

        self.g = GaiaData(self.t)

        # Use Gaia RV if not defined at dataset subclass level
        rv_name = self._radial_velocity_name

        rv = u.Quantity(self.t[rv_name])
        if rv.unit.is_equivalent(u.one):
            rv = rv * u.km/u.s
        self.c = self.g.get_skycoord(radial_velocity=rv)

    def _init_mask(self):
        # TODO: implement on subclasses
        return self.t

    def __len__(self):
        return len(self.t)

    @property
    def elem_ratios(self):
        if not hasattr(self, '_elem_ratios'):
            self._elem_ratios = ['FE_H'] + sorted([x for x in self.t.colnames
                                                   if x.endswith('_FE') and
                                                   not x.startswith('E_') and
                                                   not x.startswith('FLAG_') and
                                                   not x.startswith('CHI_') and
                                                   not x.startswith('FLUX_') and
                                                   not x.startswith('NR_')])
        return self._elem_ratios

    @property
    def elem_names(self):
        if not hasattr(self, '_elem_names'):
            elem_list = ([x.split('_')[0] for x in self.elem_ratios] +
                         [x.split('_')[1] for x in self.elem_ratios])
            elem_list.pop(elem_list.index('H'))
            self._elem_names = set(elem_list)
        return self._elem_names

    def get_elem_ratio(self, elem1, elem2=None):
        # Passed in an elem ratio provided by the table, e.g., FE_H
        if elem2 is None and elem1 in self.t.colnames:
            return self.t[elem1]

        if elem2 is None:
            try:
                elem1, elem2 = elem1.split('_')
            except Exception:
                raise RuntimeError("If passing a single elem ratio string, "
                                   "it must have the form ELEM_ELEM, not "
                                   f"{elem1}")

        elem1 = str(elem1).upper()
        elem2 = str(elem2).upper()

        if elem2 == 'H':
            i1 = self.elem_ratios.index(elem1 + '_FE')
            i2 = self.elem_ratios.index('FE_H')
            return (self.t[self.elem_ratios[i1]] -
                    self.t[self.elem_ratios[i2]])

        else:
            i1 = self.elem_ratios.index(elem1 + '_FE')
            i2 = self.elem_ratios.index(elem2 + '_FE')
            return (self.t[self.elem_ratios[i1]] -
                    self.t[self.elem_ratios[i2]])

    def get_mh_am_mask(self):
        # TODO: implement on subclasses
        return np.ones(len(self.t), dtype=bool)

    def filter(self, filters, low_alpha=True):
        mask = np.ones(len(self.t), dtype=bool)
        for k, (x1, x2) in filters.items():
            if x1 is None and x2 is None:
                raise ValueError("Doh")

            arr = u.Quantity(self.t[k]).value

            if x1 is None:
                mask &= arr < x2

            elif x2 is None:
                mask &= arr >= x1

            else:
                mask &= (arr >= x1) & (arr < x2)

        if low_alpha is not None:
            alpha_mask = self.get_mh_am_mask(low_alpha)
        else:
            alpha_mask = np.ones(len(self.t), dtype=bool)

        return self[mask & alpha_mask]

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = slice(slc, slc+1)
        return self.__class__(self.t[slc])


class APOGEEDataset(Dataset):
    _id_column = 'APOGEE_ID'
    _radial_velocity_name = 'VHELIO_AVG'
    _elem_err_fmt = '{elem_name}_ERR'

    # See: 2-High-alpha-Low-alpha.ipynb
    _mh_alpham_nodes = np.array([
        [0.6, -0.05],
        [0.6, 0.04],
        [0.15, 0.04],
        [-0.5, 0.13],
        [-0.9, 0.13],
        [-1., 0.07],
        [-0.2, -0.1],
        [0.2, -0.1],
        [0.6, -0.05]]
    )

    def _init_mask(self):
        aspcap_bitmask = np.sum(2 ** np.array([
            7,  # STAR_WARN
            23  # STAR_BAD
        ]))
        quality_mask = (
            (self.t['SNR'] > 20) &
            ((self.t['ASPCAPFLAG'] & aspcap_bitmask) == 0)
        )

        # Remove stars targeted in known clusters or dwarf galaxies:
        mask_bits = {
            'APOGEE_TARGET1': np.array([9, 18, 24, 26]),
            'APOGEE_TARGET2': np.array([10, 18]),
            'APOGEE2_TARGET1': np.array([9, 18, 20, 21, 22, 23, 24, 26]),
            'APOGEE2_TARGET2': np.array([10]),
            'APOGEE2_TARGET3': np.array([5, 14, 15])
        }
        target_mask = np.ones(len(self.t), dtype=bool)
        for name, bits in mask_bits.items():
            target_mask &= (self.t[name] & np.sum(2**bits)) == 0

        return self.t[quality_mask & target_mask]

    def get_mh_am_mask(self, low_alpha=True):
        mh_alpham_path = Path(self._mh_alpham_nodes[:-1])
        low_alpha_mask = mh_alpham_path.contains_points(
            np.stack((self.t['M_H'], self.t['ALPHA_M'])).T)

        if low_alpha:
            return low_alpha_mask

        else:
            return ((~low_alpha_mask) &
                    (self.t['M_H'] > -1) &
                    (self.t['ALPHA_M'] > 0))


class GALAHDataset(Dataset):
    _id_column = 'star_id'
    _radial_velocity_name = 'rv_galah'
    _elem_err_fmt = 'E_{elem_name}'

    # See: 2-High-alpha-Low-alpha.ipynb
    _mh_alpham_nodes = np.array([
        [0.6, -0.01],
        [0.6, 0.08],
        [0.15, 0.08],
        [-0.5, 0.17],
        [-0.9, 0.17],
        [-1., 0.11],
        [-0.2, -0.11],
        [0.2, -0.11],
        [0.6, -0.03]])

    def _init_mask(self):
        quality_mask = (
            (self.t['flag_sp'] == 0) &
            (self.t['flag_fe_h'] == 0)
        )

        # Remove stars targeted in known clusters or dwarf galaxies:
        # TODO: how to do this for GALAH??

        return self.t[quality_mask]

    def get_mh_am_mask(self, low_alpha=True):
        mh_alpham_path = Path(self._mh_alpham_nodes)
        low_alpha_mask = mh_alpham_path.contains_points(
            np.stack((np.array(self.t['FE_H']),
                      np.array(self.t['ALPHA_FE']))).T)

        if low_alpha:
            return low_alpha_mask

        else:
            return (~low_alpha) & (self.t['FE_H'] > -1)


apogee = APOGEEDataset(apogee_parent_filename)
galah = GALAHDataset(galah_parent_filename)

teff_ref = -382.5 * apogee.t['FE_H'] + 4607
rc_logg_max = 0.0018 * (apogee.t['TEFF'] - teff_ref) + 2.4

datasets = {
    'apogee-rgb-loalpha': apogee.filter({'LOGG': (1, 3.4),
                                         'TEFF': (3500, 6500),
                                         'FE_H': (-3, 1)},
                                        low_alpha=True),
    'apogee-rc-loalpha': apogee.filter({'LOGG': (1.9, rc_logg_max),
                                        'TEFF': (4200, 5400),
                                        'FE_H': (-3, 1)},
                                       low_alpha=True),
    'apogee-rgb-hialpha': apogee.filter({'LOGG': (1, 3.4),
                                         'TEFF': (3500, 6500),
                                         'FE_H': (-3, 1)},
                                        low_alpha=False),
    'apogee-ms-loalpha': apogee.filter({'LOGG': (3.75, 5),
                                        'TEFF': (5000, 6000),
                                        'FE_H': (-3, 1)},
                                       low_alpha=True),
    'galah-rgb-loalpha': galah.filter({'logg': (1, 3.5),
                                       'teff': (3500, 5500),
                                       'FE_H': (-3, 1)},
                                      low_alpha=True),
    'galah-ms-loalpha': galah.filter({'logg': (3.5, 5),
                                      'teff': (5000, 6000),
                                      'FE_H': (-3, 1)},
                                     low_alpha=True)
}

# From visual inspection of the z-vz grid plots!
elem_names = {
    'apogee-rgb-loalpha': ['FE_H', 'AL_FE', 'C_FE', 'MG_FE', 'MN_FE', 'NI_FE',
                           'N_FE', 'O_FE', 'P_FE', 'SI_FE'],
    'apogee-ms-loalpha': ['FE_H', 'AL_FE', 'C_FE', 'MG_FE', 'MN_FE', 'NI_FE',
                          'N_FE', 'O_FE', 'P_FE', 'SI_FE', 'TI_FE'],
    'galah-rgb-loalpha': ['FE_H', 'AL_FE', 'BA_FE', 'CA_FE', 'CO_FE', 'CU_FE',
                          'MG_FE', 'MN_FE', 'NA_FE', 'O_FE', 'SC_FE', 'Y_FE',
                          'ZN_FE'],
    'galah-ms-loalpha': ['FE_H', 'AL_FE', 'CA_FE', 'K_FE', 'MG_FE', 'MN_FE',
                         'NA_FE', 'SC_FE', 'TI_FE', 'Y_FE']
}
elem_names['apogee-rgb-hialpha'] = elem_names['apogee-rgb-loalpha']
elem_names['apogee-rc-loalpha'] = elem_names['apogee-rgb-loalpha']

for name in datasets:
    for path in [plot_path, cache_path]:
        this_path = path / name
        this_path.mkdir(exist_ok=True)
