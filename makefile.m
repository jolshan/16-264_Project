function [] = makefile(images, tags)
% Takes filenames and labels from getImagesAndTags.m and prints to file

% file where output will go
fid = fopen('filesAndLabels.txt', 'wt');

% Can condition on tags 1-8, each with its own probability.
% Alternatively, all conditional statements can be eliminated to include
% all files.
% Example here is roughly equal and each category has a little under 3000 examples.
for i = 1:size(tags,2)
    if tags(i) == 1
        if rand > .977
            fprintf(fid,'id1%04d/%s/%s\t %g \n',images.sp(i), images.video{i}, images.name{i}(end-8:end),  tags(i));  
        end
    end
    if tags(i) == 2
        if rand > .897
            fprintf(fid,'id1%04d/%s/%s\t %g \n',images.sp(i), images.video{i}, images.name{i}(end-8:end),  tags(i));  
        end
    end
    if tags(i) == 3 
            fprintf(fid,'id1%04d/%s/%s\t %g \n',images.sp(i), images.video{i}, images.name{i}(end-8:end),  tags(i));  
    end
    if tags(i) == 4
        if rand > .44
            fprintf(fid,'id1%04d/%s/%s\t %g \n',images.sp(i), images.video{i}, images.name{i}(end-8:end),  tags(i));  
        end
    end
end
fclose(fid);
end