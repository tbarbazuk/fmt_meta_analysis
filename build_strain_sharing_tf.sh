echo 'sample_id_1,sample_id_2,same_strain,sgb,dataset_1,dataset_2' > strain_sharing_list_allRaxml.csv

for f in pdists-filtered-normalized-predicted/*.csv
do
    sgb=$(basename "$f" .csv)

    awk -F',' -v OFS=',' -v sgb="$sgb" '
    NR > 1 {
        split($1, a, "__")
        split($2, b, "__")

        dataset1 = a[1]
        dataset2 = b[1]

        sample1 = substr($1, length(dataset1) + 3)
        sample2 = substr($2, length(dataset2) + 3)

        print sample1, sample2, $7, sgb, dataset1, dataset2
    }' "$f" >> strain_sharing_list_allRaxml.csv
done
