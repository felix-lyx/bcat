import torchdata.datapipes as dp
from data_utils import all_datasets as ds
from logging import getLogger

logger = getLogger()

ALL_DATASETS = {
    "react_diff": ds.ReactDiff2D,
    "shallow_water": ds.ShallowWater2D,
    "incom_ns": ds.IncomNS2D,
    "com_ns": ds.ComNS2D,
    "incom_ns_arena": ds.IncomNS2DArena,
    "incom_ns_arena_u": ds.IncomNS2DArenaU,
    "cfdbench": ds.CFDBench2D,
}


def get_dataset(params, symbol_env, split):
    types = params.data.types

    if split == "train":
        datasets = {}
        for t in types:
            ds = ALL_DATASETS[t](params, symbol_env, split, train=True)
            if not ds.fully_shuffled:
                # during training, shuffle iterable datasets that are not fully shuffled
                ds = ds.shuffle(buffer_size=ds.buffer_size)
            # datasets.append(ds.cycle())
            datasets[ds.cycle()] = params.data.sampler[t]
            # datasets[ds] = params.data.sampler[t]

        if params.data.sampler.uniform:
            return dp.iter.Multiplexer(*datasets)
        else:
            return dp.iter.SampleMultiplexer(datasets)
    else:

        datasets = {}
        for t in types:
            use_split = "train" if params.overfit_test else split

            ds = ALL_DATASETS[t](params, symbol_env, split=use_split, train=False)
            datasets[t] = ds

            if t == "com_ns" and params.eval_single_file:
                for i in range(8):
                    ds = ALL_DATASETS[t](params, symbol_env, split=use_split, train=False, file_idx=i)
                    file_name = (
                        ds.data_files[0].split("/")[-1].removeprefix("2D_CFD_").removesuffix("_periodic_128_Train.hdf5")
                    )
                    datasets["com_ns:{}".format(file_name)] = ds
        return datasets
