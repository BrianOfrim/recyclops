LATEST_TIMESTAMP=$(ls -1 -r ../logs/hparam_tuning/ | head -1)
NEW_FILE="${LATEST_TIMESTAMP}_notes.txt"
touch $NEW_FILE
echo $LATEST_TIMESTAMP >>  $NEW_FILE
echo "batch_size\tdropout\toptimizer\tbase_learning_rate\tdo_file_tune\tacc(Val)\tloss(val)\tacc(train)\tloss(train)" >> $NEW_FILE
echo "Created file ${NEW_FILE}"
vim $NEW_FILE
