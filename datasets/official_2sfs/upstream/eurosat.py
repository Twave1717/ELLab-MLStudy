import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

"""
template = ['a centered satellite photo of {}.']
"""
template = ['a photo of a {}.']

NEW_CNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSAT(DatasetBase):

    dataset_dir = 'eurosat'

    def __init__(self, root, num_shots, setting="standard", seed=1):
        assert setting in ("standard", "base2new")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        
        self.template = template

        # load default training, validation and test splits of the dataset
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)

        # make it a few-shot dataset by subsampling both the training and validation splits 
        # NOTE: these will be overwritten by line 33, but leaving this here s.t. you know how to create
        # a new FSL dataset on the fly if needed
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # load the preprocessed jsonl for the few-shot split according to the (seed, shots) combo
        # NOTE: u should download the shared pickles on drive first
        preprocessed_train = os.path.join(self.dataset_dir, "split_fewshot", f"shot_{num_shots}-seed_{seed}_train.jsonl")
        train = OxfordPets.load_preprocessed_jsonl(preprocessed_train)
        preprocessed_val = os.path.join(self.dataset_dir, "split_fewshot", f"shot_{num_shots}-seed_{seed}_val.jsonl")
        val = OxfordPets.load_preprocessed_jsonl(preprocessed_val)

        # subsample the classes in the sets into base/novel 
        # (for the test set, we return both base/novel test loader to compute both accs in a single script)
        if setting == "base2new":
            train, val, test_base, test_new = OxfordPets.base2new_split(train, val, test)
        elif setting == "standard":
            train, val, test_base, test_new = train, val, test, None        

        super().__init__(train_x=train, val=val, test=test_base, test_new=test_new)
    
    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(
                impath=item_old.impath,
                label=item_old.label,
                classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
