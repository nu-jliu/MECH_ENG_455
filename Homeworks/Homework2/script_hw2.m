% Allen Liu
% ME455 Active Learning
% Homework 2

close all
clear variables
clc

N = 25;
map = ones(N, N) * 1/(N^2);
xrange = [0.5;N-0.5];
yrange = [0.5;N-0.5];

dx = 1;
dy = 1;
xg = linspace(xrange(1)-dx/2,xrange(2)+dx/2,N+1);
yg = linspace(yrange(1)-dy/2,yrange(2)+dy/2,N+1);


x = [randi([1 N]); randi([1 N])];
s = [randi([1 N]); randi([1 N])];


start = x;
traj = [];

fig1 = figure(Position=[200 200 1000 1000]);
ax1 = axes(fig1);
set(ax1, 'YDir', 'normal')
set(ax1, 'XLim', [0 25])
set(ax1, 'YLim', [0 25])
hold(ax1, 'on')
axis(ax1, 'equal')

fig2 = figure(Position=[200, 200, 2000, 1000]);

set(fig1, 'Color', 'w')
set(fig2, 'Color', 'w')

plotInd = 1;
for i = 1:100
    traj(:, i) = x - [0.5; 0.5];
    measure = likelihood(s, x, true);
    z = rand() < measure;
    
    map = update_map(map, x, z);
    x = move_random(x);

    cla(ax1)
    imagesc(ax1, xrange, yrange, map)
    colormap(ax1, 'turbo')
    colorbar(ax1)
    mesh(ax1, xg, yg, zeros([N+1 N+1]), FaceColor='none', EdgeColor='k', DisplayName='Grid')
    plot(ax1, traj(1, :), traj(2, :), LineStyle='-', Marker='.', DisplayName='Robot Path')
    plot(ax1, s(1)-.5, s(2)-.5, LineStyle='none', Marker='x', LineWidth=2, MarkerSize=10, DisplayName='Source Location')
    plot(ax1, start(1)-.5, start(2)-.5, LineStyle='none', Marker='x', LineWidth=2, MarkerSize=10, DisplayName='Start Location')
    legend(ax1, 'show')
    drawnow

    if rem(i, 10) == 0
        ax2 = subplot(2, 5, plotInd, 'Parent', fig2);

        hold(ax2, 'on')
        imagesc(ax2, xrange, yrange, map)
        colormap(ax2, 'turbo')
        colorbar(ax2)
        mesh(ax2, xg, yg, zeros([N+1 N+1]), FaceColor='none', EdgeColor='k', DisplayName='Grid')
        plot(ax2, traj(1, :), traj(2, :), LineStyle='-', Marker='.', DisplayName='Robot Path')
        plot(ax2, s(1)-.5, s(2)-.5, LineStyle='none', Marker='x', LineWidth=2, MarkerSize=10, DisplayName='Source Location')
        plot(ax2, start(1)-.5, start(2)-.5, LineStyle='none', Marker='x', LineWidth=2, MarkerSize=10, DisplayName='Start Location')
        hold(ax2, 'off')

        set(ax2, 'YDir', 'normal')
        set(ax2, 'XLim', [0 25])
        set(ax2, 'YLim', [0 25])
        axis(ax2, 'equal')

        legend(ax2, 'show')

        plotInd = plotInd + 1;
    end
end

hold(ax1, 'off')

saveas(fig2, 'part1.png')

function l = likelihood(s, x, z)
    diff = s - x;
    xDiff = abs(diff(1));
    yDiff = abs(diff(2));

    if yDiff == 3
        if xDiff <= 3
            l = 1/4;
        else
            l = 0;
        end
    elseif yDiff == 2
        if xDiff <= 2
            l = 1/3;
        else
            l = 0;
        end
    elseif yDiff == 1
        if xDiff <= 1
            l = 1/2;
        else
            l = 0;
        end
    elseif yDiff == 0
        if xDiff == 0
            l = 1;
        else
            l = 0;
        end
    else
        l = 0;
    end

    if ~z
        l = 1 - l;
    end
end

function map = update_map(map, x, z)
    row = size(map, 1);
    col = size(map, 2);

    px = 0;
    dx = 1/(row*col);

    for i = 1:row
        for j = 1:col
            s = [j; i];
            pzx = likelihood(s, x, z);
            bx = map(i, j);
            px = px + pzx*bx*dx;
        end
    end

    b_total = 0;

    for i = 1:row
        for j = 1:col
            s = [j; i];
            bx = map(i, j);
            pzx = likelihood(s, x, z);
            bx_new = pzx*bx/px;
            b_total = b_total + bx_new;
            map(i, j) = bx_new;
        end
    end

    for i = 1:row
        for j = 1:col
            map(i, j) = map(i, j) / b_total;
        end
    end
end

function v = isValid(x)
    x1 = x(1);
    x2 = x(2);

    v = x1 > 0 && x1 < 26 && x2 > 0 && x2 < 26;
end

function x = move_random(x)
    i = 0;
    nextStep = [];
    
    x1 = x(1);
    x2 = x(2);

    up = [x1; x2+1];
    down = [x1; x2-1];
    left = [x1-1; x2];
    right = [x1+1; x2];

    if isValid(up)
        i = i + 1;
        nextStep(:, i) = up;
    end

    if isValid(down)
        i = i + 1;
        nextStep(:, i) = down;
    end

    if isValid(left)
        i = i + 1;
        nextStep(:, i) = left;
    end

    if isValid(right)
        i = i + 1;
        nextStep(:, i) = right;
    end

    randInd = randi([1 i]);

    % disp(size(nextStep))
    % disp(i)
    % disp(randInd)

    x = nextStep(:, randInd);
end

function x = move(map, x)
    x1 = x(1);
    x2 = x(2);

    up = [x1; x2+1];
    down = [x1; x2-1];
    left = [x1-1; x2];
    right = [x1+1; x2];

    p = 0;

    if isValid(up)
        b = map(up(2), up(1));
        if b > p
            p = b;
            x = up;
        end
    end

    if isValid(down)
        b = map(down(2), down(1));
        if b > p
            p = b;
            x = down;
        end
    end

    if isValid(left)
        b = map(left(2), left(1));
        if b > p
            p = b;
            x = left;
        end
    end

    if isValid(right)
        b = map(right(2), right(1));
        if b > p
            p = b;
            x = right;
        end
    end
end