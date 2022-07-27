`merge_json.py` can be used to combine two JSON files which follow the structure similar to the one specified [here](https://wadhwaniai.atlassian.net/wiki/spaces/CF/pages/2287075392/ML+Data+Format). Merging can be configured by specifying:

1. Strategy to be used for merging (`keep_first`, `keep_second`, `extend`, `unique_extend`)
2. Which combination of keys have a unique value in each table
3. How keys in one table reference keys in another table
4. If prefix has to be added to values of either of the files.

Here table refers to a list of dictionaries. Keys of the dictionaries form the columns and values in each dictionary become a row in the table.

# Usage
```
python merge_json.py -h
usage: merge_json.py [-h] -f FIRST_FILEPATH -s SECOND_FILEPATH
                     [-d DEST_FILEPATH] -c CONFIG_PATH

optional arguments:
  -h, --help            show this help message and exit
  -f FIRST_FILEPATH, --first_filepath FIRST_FILEPATH
                        path to first JSON file
  -s SECOND_FILEPATH, --second_filepath SECOND_FILEPATH
                        path to second JSON file
  -d DEST_FILEPATH, --dest_filepath DEST_FILEPATH
                        path to file where merged JSON will be saved. If None,
                        JSON is dumped to stdout.
  -c CONFIG_PATH, --config_path CONFIG_PATH
                        path to merging config
```
## Suggested Usage
Merge counting model's JSON and image validation model's JSON:
```
python merge_json.py -f <path to counting JSON> -s <path to validation JSON> -d <path to destination file> -c configs/merge_config_count_val.yaml
```
Merge Ground Truth JSON and Prediction JSON:
```
python merge_json.py -f <path to GT JSON> -s <path to Pred JSON> -d <path to destination file> -c configs/merge_config_gt_pred.yaml
```

# Config
Merging config contains 5 sections for each table:
1. **id_key**: name of the key which is serves as an index in current table. Indexes have to be reset after merging hence they are treated differently.
2. **unique_keys**: list of keys which are unque for each row
3. **references**: information about how keys in current table reference other table
4. **prefix**: information about how values in the current table have to be changed before merging
5. **merge_strategy**: information about how to merge

## General structure
```
schema:
  <table name 1>:                                                            # name of the table
    id_key: <name of the key which serves as index in current table>
    unique_keys:                                                             # list of keys which when considered together are supposed to be unique for each row
      - <name of key 1>
      - <name of key 2>
      .
      .
    references:                                                              # details about how keys in current table have foreign key dependency on keys of other table
      <name of the key in current table which references other table>:
        - <name of the referenced table>
        - <name of the referenced key>
      .
      .
    prefix:                                                                  # details about how values in the current table have to be changed before merging
      <name of the key whose values will be changed>:
        first: <prefix to be added in first file>
        second: <prefix to be added in second file>
      .
      .
    merge_strategy: <which strategy to use for merging>                      # has to be one of the implemented strategy
  .
  .
```
## Example config
```
schema:
  info: {}                             # `info` is an empty dict so it won't be affected (alternatively we can skip mentioning `info` here)
  images:
    id_key: id                         # key named `id` is index in `images` table (indexes are treated differently than other keys as they are reset after merging)
    unique_keys:                       # `file_path` alone is unique in all rows of the `images` table
      - file_path
    merge_strategy: unique_extend      # `images` has to be merged using `unique_extend` strategy
  categories:
    id_key: id                         # key named `id` is index in categories table
    unique_keys:                       # `name` and `supercategory` when taken together are unique in all rows of categories table
      - name
      - supercategory
    merge_strategy: unique_extend
    prefix:
      supercategory:                   # add prefix to `supercategory` key in the categories table
        first: null                    # dont't add a prefix in first file (alternatively we can skip mentioning `first`)
        second: predicted_             # add `predicted_` prefix to supercategories in second file
  box_annotations:
    id_key: id
    unique_keys:
      - image_id
      - category_id
      - bbox
      - bbox_score
    references:
      image_id:                        # `image_id` key in `box_annotations` references `id` key in `images` table
        - images
        - id
      category_id:                     # `category_id` key in `box_annotations` references `id` key in `categories` table
        - categories
        - id
    merge_strategy: extend
  caption_annotations:
    id_key: id
    unique_keys:
      - image_id
      - category_id
      - caption
      - conf
    references:
      image_id:
        - images
        - id
      category_id:
        - categories
        - id
    merge_strategy: extend
  splits:
    unique_keys:
      - image_id
      - split
    references:
      image_id:
        - images
        - id
    merge_strategy: keep_first
```
# Merge Strategies
1. **keep_first**: select the table from the first file  (might be useful for merging `splits` table)
2. **keep_second**: select the table from the second file  (similar to keep_first)
3. **extend**: append rows from both files  (might be useful for merging `box_annotations` or `caption_annotation`)
4. **unique_extend**: append rows from both files but drop duplicate rows (might be useful for merging `images` table)

# Test
[test_json_merged.json](./tests/resources/test_json_merged.json) is produced by merging [test_json_1.json](./tests/resources/test_json_1.json) and [test_json_2.json](./tests/resources/test_json_2.json) using this [merge config](./tests/resources/merge_config.yaml) by running the following command:
```
python merge_json.py -f tests/resources/test_json_1.json -s tests/resources/test_json_2.json -c tests/resources/merge_config.yaml -d tests/resources/test_json_merged.json
```
