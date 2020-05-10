function [] = makefileCSV(images, tags)
% Takes filenames and labels from getImagesAndTags.m and prints to file
% Used for audioCNN2.py only as it requires csv format.

fid1 = fopen('filesAndLabelsTrain.csv', 'wt');
fid2 = fopen('filesAndLabelsTest.csv', 'wt');
fprintf(fid1, 'ID,Class\n');
fprintf(fid2, 'ID,Class\n');

% Only use labels 1-5
for i = 1:size(tags,2)
    if tags(i) ~= 6 && tags(i) ~= 7 && tags(i) ~= 8
        if rand > .10
            fprintf(fid1,'wav/id1%04d/%s/%s,%g\n',images.sp(i), images.video{i}, images.name{i}(end-8:end),  tags(i));
        else
            fprintf(fid2,'wav/id1%04d/%s/%s,%g\n',images.sp(i), images.video{i}, images.name{i}(end-8:end),  tags(i));  
        end
    end
end

fclose(fid1);
fclose(fid2);
end