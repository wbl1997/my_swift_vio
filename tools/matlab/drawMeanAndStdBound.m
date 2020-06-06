function line_handles = drawMeanAndStdBound(data, meanIndex, ...
    stdIndex, scale, isError, meanLineStyles, stdLineStyles)
% data 
% isError is the mean an error? If so, we will plot 3 * std; 
% otherwise, we will plot 3 * std + mean
% dimen should be smaller than 10.
if nargin < 7
    stdLineStyles = {'--r', '--g', '--b', '--k', '-.k', '-.b', ...
    '--c', '--m', '--y'};
end
if nargin < 6
    meanLineStyles = {'-r', '-g', '-b', '-k', '.k', '.b', ...
        '-c', '-m', '-y'};
end
if nargin < 5
    isError = 0;
end
if nargin < 4
    scale = 1.0;
end

dimen = length(meanIndex);
line_handles = zeros(1, dimen * 3);
line_handles(1:dimen) = drawColumnsInMatrix(...
    data, meanIndex, scale, 0, meanLineStyles);

if isError
    for i=1:dimen
        line_handles(dimen + i) = plot(data(:,1), ...
            (3*data(:, stdIndex(i)))*scale, stdLineStyles{i});
        line_handles(dimen*2 + i) = plot(data(:,1), ...
            (-3*data(:,stdIndex(i)))*scale, stdLineStyles{i});
    end
else
    for i=1:dimen
        line_handles(dimen + i) = plot(data(:,1), ...
            (3*data(:, stdIndex(i)) + ...
            data(:, meanIndex(i)))*scale, stdLineStyles{i});
        line_handles(dimen*2 + i) = plot(data(:,1), ...
            (-3*data(:,stdIndex(i)) + ...
            data(:, meanIndex(i)))*scale, stdLineStyles{i});
    end
end

legendLabels = cell(dimen * 2, 1);
if dimen <= 3
    candidateLegendLabels = {'x', 'y', 'z'};
else
    candidateLegendLabels = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};   
end
for i = 1:dimen
    legendLabels{i} = candidateLegendLabels{i};
    legendLabels{dimen + i} = ['3\sigma_', candidateLegendLabels{i}];
end
legend(legendLabels);

xlabel('time (sec)');
grid on;
end

