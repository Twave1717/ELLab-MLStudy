import os
from scipy.io import loadmat

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase


template = ['a photo of a {}.']


class StanfordCars(DatasetBase):

    dataset_dir = 'stanford_cars'

    def __init__(self, root, num_shots, setting="standard", seed=1):
        assert setting in ("standard", "base2new")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_StanfordCars.json')

        self.template = template

        # load default training, validation and test splits of the dataset
        train, val, test = OxfordPets.read_split(self.split_path, self.dataset_dir)

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
        

        print(f"Number of Train, Val and Test classes = {len(self.classnames)}, {len(self.val_classnames)} and {len(self.test_classnames)}")
        if setting == "base2new":
            assert all([ct == cv for ct, cv in zip(self.classnames, self.val_classnames)]) # ensure train and val classes are the same
            assert not any([ct == cv for ct, cv in zip(self.classnames, self.test_new_classnames)]) # ensure train and test classes are different
    
    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)['annotations'][0]
        meta_file = loadmat(meta_file)['class_names'][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]['fname'][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]['class'][0, 0]
            label = int(label) - 1 # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            classname = ' '.join(names)
            item = Datum(
                impath=impath,
                label=label,
                classname=classname
            )
            items.append(item)
        
        return items