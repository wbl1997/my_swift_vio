function compute_stats_from_file(data_file, indices, output_file)
% compute stats of columns of indices in data_file
% write the mean, std, and max to output_file in append mode
data = readmatrix(data_file, 'NumHeaderLines', 1);
m = mean(data(:, indices));
x = max(data(:, indices));
s = std(data(:, indices));
fileID = fopen(output_file, 'a');
fprintf(fileID, '%s\n', data_file);
fprintf(fileID, 'mean: ');
for j = 1:length(indices)
    fprintf(fileID, '%f\t', m(j));
end
fprintf(fileID, '\n');
fprintf(fileID, 'std: ');
for j = 1:length(indices)
    fprintf(fileID, '%f\t', s(j));
end

fprintf(fileID, '\n');
fprintf(fileID, 'max: ');
for j = 1:length(indices)
    fprintf(fileID, '%f\t', x(j));
end
fprintf(fileID, '\n\n');
fclose(fileID);
end

