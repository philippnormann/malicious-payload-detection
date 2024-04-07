#!/bin/bash

kaggle_datasets=("syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning")

github_repos=("https://github.com/payloadbox/xss-payload-list" 
              "https://github.com/msudol/Web-Application-Attack-Datasets" 
              "https://github.com/grananqvist/Machine-Learning-Web-Application-Firewall-and-Dataset")

data_folder="data"
if [ ! -d "$data_folder" ]; then
    mkdir "$data_folder"
fi

pushd "$data_folder"

    # Download the Kaggle datasets
    for dataset in "${kaggle_datasets[@]}"; do
        kaggle datasets download --force -d "$dataset"
        dataset_name=$(basename "$dataset")
        unzip -o "$dataset_name.zip" -d "$dataset_name"
        rm "$dataset_name.zip"
    done

    # Clone the GitHub repos
    for repo in "${github_repos[@]}"; do
        repo_name=$(basename "$repo" .git)
        if [ -d "$repo_name" ]; then
            rm -rf "$repo_name"
        fi
        git clone "$repo"
    done

popd