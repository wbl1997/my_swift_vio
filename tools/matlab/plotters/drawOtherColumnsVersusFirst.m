function drawOtherColumnsVersusFirst(data, colorstring, linespec)
% plot the second to last columns relative to the first column as x axis
% data rows x (1+n), n <= 4
% for each of the second to last column, the style is chosen from 
% colorstring x linespec
% 
% Example: 
% colorspec = {[0.9 0.9 0.9]; [0.8 0.8 0.8]; [0.6 0.6 0.6]; ...
%   [0.4 0.4 0.4]; [0.2 0.2 0.2]};
% colorstring = 'kbgry';
% linespec = {'-', '-', ':', '-.', '--'};
% bundle_plot(data, colorstring, linespec);

hold on
for i = 2 : size(data, 2)
  colorid = mod(i - 2, length(colorstring)) + 1;
  lineid = floor((i - 2)/length(colorstring)) + 1;
  plot(data(:, 1), data(:, i), 'Color', colorstring(colorid), ...
      'LineStyle', linespec{lineid});
end
