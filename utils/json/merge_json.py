import argparse
import datetime
import getpass
import json
import os
import uuid
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple, Union

from omegaconf import OmegaConf


class Merge:
    """Class to define different ways in which two list of dictionaries can be merged."""

    def __init__(self):
        """Creates mapping from string to function pointer"""
        self.string_to_function = {
            "keep_first": self.keep_first,
            "keep_second": self.keep_second,
            "extend": self.extend,
            "unique_extend": self.unique_extend,
        }

    def merge(
        self,
        list_of_dicts_first: List[Dict],
        list_of_dicts_second: List[Dict],
        strategy: str,
        **kwargs,
    ) -> List[Dict]:
        """Merges two list of dictionaries using the specified merge strategy

        Parameters
        ----------
        list_of_dicts_first : List[Dict]
            first list of dictionaries
        list_of_dicts_second : List[Dict]
            second list of dictionaries
        strategy : str
            name of merge strategy to be used

        Returns
        -------
        List[Dict]
            merged list of dicitonaries
        """
        list_of_dicts_first = deepcopy(list_of_dicts_first)
        list_of_dicts_second = deepcopy(list_of_dicts_second)
        function = self.string_to_function.get(strategy)
        assert (
            function is not None
        ), f"{strategy} merging strategy needs to be one of {self.string_to_function.keys()}"
        return function(list_of_dicts_first, list_of_dicts_second, **kwargs)

    def keep_first(
        self, list_of_dicts_first: List[Dict], list_of_dicts_second: List[Dict], **kwargs
    ) -> List[Dict]:
        """Returns the first list of dictionaries

        Parameters
        ----------
        list_of_dicts_first : List[Dict]
            first list of dictionaries
        list_of_dicts_second : List[Dict]
            second list of dictionaries

        Returns
        -------
        List[Dict]
            first list of dictionaries
        """
        return list_of_dicts_first

    def keep_second(
        self, list_of_dicts_first: List[Dict], list_of_dicts_second: List[Dict], **kwargs
    ) -> List[Dict]:
        """Returns the second list of dictionaries

        Parameters
        ----------
        list_of_dicts_first : List[Dict]
            first list of dictionaries
        list_of_dicts_second : List[Dict]
            second list of dictionaries

        Returns
        -------
        List[Dict]
            second list of dictionaries
        """
        return list_of_dicts_second

    def extend(
        self, list_of_dicts_first: List[Dict], list_of_dicts_second: List[Dict], **kwargs
    ) -> List[Dict]:
        """Returns union of input list of dictionaries

        Parameters
        ----------
        list_of_dicts_first : List[Dict]
            first list of dictionaries
        list_of_dicts_second : List[Dict]
            second list of dictionaries

        Returns
        -------
        List[Dict]
            extended list of dictionaries
        """
        return list_of_dicts_first + list_of_dicts_second

    def unique_extend(
        self,
        list_of_dicts_first: List[Dict],
        list_of_dicts_second: List[Dict],
        unique_keys: Iterable,
        **kwargs,
    ) -> List[Dict]:
        """Returns deduplicated union of input list of dicitonaries

        Parameters
        ----------
        list_of_dicts_first : List[Dict]
            first list of dictionaries
        list_of_dicts_second : List[Dict]
            second list of dictionaries
        unique_keys : Iterable
            keys on which duplicates will be checked

        Returns
        -------
        List[Dict]
            extended list of dictionaries with duplicate dictionaries removed
        """
        assert (
            unique_keys is not None
        ), "unique_keys are required for unique_extend merging strategy"
        merged_list = list_of_dicts_first + list_of_dicts_second
        merged_list = check_duplicates(merged_list, on_keys=unique_keys, drop_duplicates=True)
        return merged_list


def add_prefix(list_of_dicts: List[Dict], id_key: str, prefix: str) -> List[Dict]:
    """Adds prefix to the values of `id_key` key in each dictionary of list of dictionaries.

    Parameters
    ----------
    list_of_dicts : List[Dict]
        list of dictionaries. Each dictionary in the list should have `id_key` key
    id_key : str
        key whose values are to be modified
    prefix : str
        prefix to be added

    Returns
    -------
    List[Dict]
        list of dictionaries with modified values of `id_key` key
    """
    list_of_dicts = deepcopy(list_of_dicts)
    for d in list_of_dicts:
        d[id_key] = f"{prefix}{d[id_key]}"
    return list_of_dicts


def replace_id_with_value(
    list_of_dicts: List[Dict], id_key: str, id_to_value_map: Dict
) -> List[Dict]:
    """Replaces values of `id_key` key using a current_value to final_value mapping

    Parameters
    ----------
    list_of_dicts : List[Dict]
        list of dictionaries. Each dictionary in the list should have `id_key` key
    id_key : str
        key whose values are to be modified
    id_to_value_map : Dict
        current_value to final_value mapping

    Returns
    -------
    List[Dict]
        list of dictionaries with modified values of `id_key` key
    """
    list_of_dicts = deepcopy(list_of_dicts)
    for d in list_of_dicts:
        d[id_key] = id_to_value_map[d[id_key]]
    return list_of_dicts


def check_duplicates(
    list_of_dicts: List[Dict], on_keys: Iterable, drop_duplicates: bool = True
) -> List[Dict]:
    """Checks for duplicate dictionaries based on `on_keys` in a list of dictionaries. Removes
    duplicates if `drop_duplicates` is True.

    Parameters
    ----------
    list_of_dicts : List[Dict]
        list of dictionaries
    on_keys : Iterable
        keys on which duplicates will be checked
    drop_duplicates : bool, optional
        flag whether to remove duplicates, by default True

    Returns
    -------
    List[Dict]
        list of dictionaries
    """
    list_of_dicts = deepcopy(list_of_dicts)
    seen_values = set()
    deduplicated_list = []
    for d in list_of_dicts:
        values = [d.get(x) for x in on_keys]
        values = tuple(
            [tuple(x) if isinstance(x, list) else x for x in values]
        )  # convert list to tuple because list is unhashable
        if values not in seen_values:
            deduplicated_list.append(d)
            seen_values.add(values)

    if drop_duplicates:
        return deduplicated_list
    else:
        print(
            f"{len(list_of_dicts) - len(deduplicated_list)} duplicate entries found on following"
            f" keys: {on_keys}"
        )
        return list_of_dicts


def reset_key(list_of_dicts: List[Dict], key: str, fill_value: List) -> List[Dict]:
    """Resets values of the `key` key by the values passed in `fill_value`

    Parameters
    ----------
    list_of_dicts : List[Dict]
        list of dictionaries
    key : str
        key whose values are to be modified
    fill_value : List
        list of values which will replace the original values

    Returns
    -------
    List[Dict]
        list of dictionaries with modified values of `key` key
    """
    list_of_dicts = deepcopy(list_of_dicts)
    for idx, d in enumerate(list_of_dicts):
        d[key] = fill_value[idx]
    return list_of_dicts


def parse_merging_config(config: Dict) -> Tuple:
    """Extracts relevant information from merging config

    Parameters
    ----------
    config : Dict
        merging config

    Returns
    -------
    Tuple
        `(schema, unique_keys, merge_strategy, references, prefix)`
    """
    schema = config["schema"]
    unique_keys = {
        k: v["unique_keys"] for k, v in config["schema"].items() if "unique_keys" in v.keys()
    }
    merge_strategy = {
        k: v["merge_strategy"] for k, v in config["schema"].items() if "merge_strategy" in v.keys()
    }
    references = {
        k: v["references"] for k, v in config["schema"].items() if "references" in v.keys()
    }
    prefix = {k: v["prefix"] for k, v in config["schema"].items() if "prefix" in v.keys()}

    return schema, unique_keys, merge_strategy, references, prefix


def merge_jsons(
    first_file: Union[str, Dict],
    second_file: Union[str, Dict],
    config_path: str,
    dest_filepath: Optional[str] = None,
) -> Dict:
    """
    Methodology:
    Following steps happen in order to merge two JSONs:
        STEP 1: Add prefix to differentiate between first file and second file (for example, add
        'predicted_' prefix to supercategory in one of the file).
        STEP 2: One table references other table using index. Merging two JSONs can affect the
        indexes, hence, replace referenced values with true values (for example, replace image_id in
        box_annotations with file_path).
        STEP 3: Merge tables according to their merging strategy.
        STEP 4: Reset index which gets affected because of merging.
        STEP 5: Replace true values with referenced values (opposite of STEP 2).
        STEP 6: Create info.
        STEP 7: Combine everything and save.

    Parameters
    ----------
    first_file : Union[str, Dict]
        first JSON as either dict of filepath
    second_file : Union[str, Dict]
        second JSON as either dict of filepath
    config_path : str
        path to merging config
    dest_filepath : Optional[str], optional
        if not None, path where merged json will be saved, by default None

    Returns
    -------
    Dict
        merged JSON
    """
    # If file_paths are passes, read JSONs
    if isinstance(first_file, str):
        with open(first_file) as f:
            first_file = json.load(f)

    if isinstance(second_file, str):
        with open(second_file) as f:
            second_file = json.load(f)

    config = OmegaConf.load(config_path)

    schema, unique_keys, merge_strategy, references, prefix = parse_merging_config(config)

    # STEP 1: add prefix
    for table, prefix_details in prefix.items():
        for key, prefixes in prefix_details.items():
            if prefixes.get("first") is not None:
                first_file[table] = add_prefix(first_file[table], key, prefixes.get("first"))
            if prefixes.get("second") is not None:
                second_file[table] = add_prefix(second_file[table], key, prefixes.get("second"))

    # STEP 2: replace reference values by true values so that indexes can be manipulated freely
    for file in [first_file, second_file]:
        reference_map = {}  # dict to store reference_value:true_value mapping
        for table, reference_details in references.items():
            for key, referenced_item in reference_details.items():
                referenced_table = referenced_item[0]
                referenced_key = referenced_item[1]
                referenced_table_unique_keys = unique_keys.get(referenced_table)
                assert referenced_table_unique_keys is not None, (
                    "Unique keys are required for {referenced_table} because it is referenced by"
                    " {table}."
                )

                dict_name = f"{key}_to_{'_'.join(referenced_table_unique_keys)}"
                if dict_name not in reference_map.keys():
                    reference_map[dict_name] = {
                        d[referenced_key]: tuple([d[k] for k in referenced_table_unique_keys])
                        for d in file[referenced_table]
                    }
                file[table] = replace_id_with_value(file[table], key, reference_map[dict_name])

    # STEP 3 and 4
    # merge tables
    merger = Merge()
    merged_file = {}
    for table, strategy in merge_strategy.items():
        merged_file[table] = merger.merge(
            first_file[table],
            second_file[table],
            strategy,
            unique_keys=unique_keys.get(table),
        )
        # reset index key which might have been affected because of merging
        id_key = schema[table].get("id_key")
        if id_key is not None:
            merged_file[table] = reset_key(
                merged_file[table], id_key, list(range(len(merged_file[table])))
            )

    # STEP 5: now that the indexes have been reset, replace true values with reference values
    reference_map = {}  # dict to store reference_value:true_value mapping
    for table, reference_details in references.items():
        for key, referenced_item in reference_details.items():
            referenced_table = referenced_item[0]
            referenced_key = referenced_item[1]
            referenced_table_unique_keys = unique_keys.get(referenced_table)

            dict_name = f"{'_'.join(referenced_table_unique_keys)}_to_{key}"
            if dict_name not in reference_map.keys():
                reference_map[dict_name] = {
                    tuple([d[k] for k in referenced_table_unique_keys]): d[referenced_key]
                    for d in merged_file[referenced_table]
                }
            merged_file[table] = replace_id_with_value(
                merged_file[table], key, reference_map[dict_name]
            )

    # STEP 6: create info table
    info = {}
    info["version"] = str(uuid.uuid4())
    info["description"] = json.dumps(
        {
            "first": first_file["info"],
            "second": second_file["info"],
            "merge_config": os.path.abspath(config_path),
        }
    )
    info["contributor"] = getpass.getuser()
    info["url"] = None if dest_filepath is None else os.path.abspath(dest_filepath)
    info["date_created"] = str(datetime.date.today())

    # STEP 7: combine everything and save file
    merged_file["info"] = info

    if dest_filepath is not None:
        with open(dest_filepath, "w") as f:
            json.dump(merged_file, f)

    return merged_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--first_filepath", required=True, help="path to first JSON file")
    parser.add_argument("-s", "--second_filepath", required=True, help="path to second JSON file")
    parser.add_argument(
        "-d",
        "--dest_filepath",
        default=None,
        help="path to file where merged JSON will be saved. If None, JSON is dumped to stdout.",
    )
    parser.add_argument("-c", "--config_path", required=True, help="path to merging config")
    args = parser.parse_args()

    merged_file = merge_jsons(
        args.first_filepath, args.second_filepath, args.config_path, args.dest_filepath
    )

    if args.dest_filepath is None:
        print(merged_file)
