% Allen Liu
% ME455 Active Learning
% Homework 1

close all
clear variables
clc

% figure
% hold on

s = [0.3; 0.4];
N = 1e2;
map = zeros(N+1, N+1);

%% Visualization
n = 0;
for i = 1:N+1
    for j = 1:N+1
        x = [(i-1)/N; (j-1)/N];

        f_val = positivity(x, s);
        % display(f_val);
        n = n + 1;

        map(j, i) = f_val;
    end
end

%% Part 1
positive = [];
negative = [];

posInd = 1;
negInd = 1;

for i = 1:100
    x_val = rand();
    y_val = rand();
    f_val = positivity([x_val; y_val], s);

    measure = rand();
    if (measure < f_val)
        positive(:, posInd) = [x_val; y_val];
        posInd = posInd + 1;
    else
        negative(:, negInd) = [x_val; y_val];
        negInd = negInd + 1;
    end
end

figure
hold on

xlim = [0;1];
ylim = [0;1];
imagesc(xlim, ylim, map)
colormap gray
colorbar

plot(0.3, 0.4, MarkerSize=20, LineStyle='none', Marker='x', DisplayName='Source')
plot(positive(1, :), positive(2, :), LineStyle='none', Color='g', Marker='.', DisplayName='Positive Signals')
plot(negative(1, :), negative(2, :), LineStyle='none', Color='r', Marker='.', DisplayName='Negative Signals')
hold off

grid minor
legend show
title('Measurements for 100 Samples')

set(gca, 'YDir', 'normal')
set(gca, 'XLim', [0 1])
set(gca, 'YLim', [0 1])
set(gcf, 'Color', 'w')

saveas(gcf, 'part1.png')

%% Part 2
map_est = zeros(N+1, N+1);
for i = 1:N+1
    for j = 1:N+1
        x_val = (i-1)/N;
        y_val = (j-1)/N;

        map_est(i, j) = l_source([x_val; y_val], positive, negative);
    end
end

figure
hold on

imagesc(xlim, ylim, map_est)
colormap gray
colorbar

plot(positive(1, :), positive(2, :), Color='g', LineStyle='none', Marker='.', DisplayName='Positive Signals')
plot(negative(1, :), negative(2, :), Color='r', LineStyle='none', Marker='.', DisplayName='Negative Signals')

hold off
legend show
title('Likelyhood of Source Location')

set(gca, 'XLim', [0 1])
set(gca, 'YLim', [0 1])
set(gcf, 'Color', 'w')

saveas(gcf, 'part2.png')


%% Part 3
positive = [];
negative = [];

posInd = 1;
negInd = 1;

val = rand();

for i = 1:100
    x_val = val;
    y_val = val;
    f_val = positivity([x_val; y_val], s);

    measure = rand();
    if (measure < f_val)
        positive(:, posInd) = [x_val; y_val];
        posInd = posInd + 1;
    else
        negative(:, negInd) = [x_val; y_val];
        negInd = negInd + 1;
    end
end

map_est = zeros(N+1, N+1);
for i = 1:N+1
    for j = 1:N+1
        x_val = (i-1)/N;
        y_val = (j-1)/N;

        map_est(i, j) = l_source([x_val; y_val], positive, negative);
    end
end

figure
hold on

imagesc(xlim, ylim, map_est)
colormap gray
colorbar

if size(positive, 2) > 0
    plot(positive(1, :), positive(2, :), DisplayName='Positive Signals', LineStyle='none', Color='g', Marker='.')
end

if size(negative, 2) > 0
    plot(negative(1, :), negative(2, :), DisplayName='Negative Signals', LineStyle='none', Color='r', Marker='.')
end

hold off
legend show
title('Likelyhood of Source Location')

set(gca, 'XLim', [0 1])
set(gca, 'YLim', [0 1])
set(gcf, 'Color', 'w')

saveas(gcf, 'part3.png')

%% Part 4
figure(Position=[200 200 2000 800])
x = [rand(); rand()];


for i = 1:10
    subplot(2, 5, i)
    map = ones(N+1, N+1);
    imagesc(xlim, ylim, map)
    colormap gray
    colorbar

    set(gca, 'XLim', [0 1])
    set(gca, 'YLim', [0 1])
end

set(gcf, 'Color', 'w')
saveas(gcf, 'part4.png')

%% Helper Functions
function f_x = positivity(x, s)
    f_x = exp(-100*(norm(x-s)-0.2)^2);
end

function p_x = l_reading(x, s, z)
    if z
        p_x = exp(-100*(norm(x-s)-0.2)^2);
    else
        p_x = 1-exp(-100*(norm(x-s)-0.2)^2);
    end
end

function l_s = l_source(s, pos, neg)
    l_s = 1;
    n_pos = size(pos, 2);
    n_neg = size(neg, 2);

    for i = 1:n_pos
        x = pos(:, i);
        l_s = l_s*l_reading(x, s, true);
    end

    for i = 1:n_neg
        x = neg(:, i);
        l_s = l_s*l_reading(x, s, false);
    end

end

function map = update_map(map, x, measure)
    row = size(map, 1);
    col = size(map, 2);
    
    for i = 1 : row
        for j = 1 : col
            s = [(j-1)/100; (i-1)/100];
        end
    end
end