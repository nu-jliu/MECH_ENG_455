% Allen Liu
% ME455 Active Learning for Robotics
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

%% Part 1
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
    traj = [traj x - [0.5; 0.5]];
    measure = likelihood(s, x, true);
    z = rand() < measure;
    
    map = update_map(map, x, z);
    x = move_random(x);
    % move(map, x);

    cla(ax1)
    imagesc(ax1, xrange, yrange, map)
    colormap(ax1, 'hot')
    colorbar(ax1)
    mesh(ax1, xg, yg, zeros([N+1 N+1]), FaceColor='none', EdgeColor='k', DisplayName='Grid')
    plot(ax1, traj(1, :), traj(2, :), LineStyle='-', Marker='.', DisplayName='Robot Path')
    plot(ax1, s(1)-.5, s(2)-.5, LineStyle='none', Marker='x', LineWidth=2, MarkerSize=10, DisplayName='Source Location')
    plot(ax1, start(1)-.5, start(2)-.5, LineStyle='none', Marker='.', LineWidth=2, MarkerSize=10, DisplayName='Start Location')
    legend(ax1, 'show')
    drawnow

    if rem(i, 10) == 0 && plotInd <= 10
        ax2 = subplot(2, 5, plotInd, 'Parent', fig2);

        hold(ax2, 'on')
        imagesc(ax2, xrange, yrange, map)
        colormap(ax2, 'hot')
        colorbar(ax2)
        mesh(ax2, xg, yg, zeros([N+1 N+1]), FaceColor='none', EdgeColor='k', DisplayName='Grid')
        plot(ax2, traj(1, :), traj(2, :), LineStyle='-', Marker='.', DisplayName='Robot Path')
        plot(ax2, s(1)-.5, s(2)-.5, LineStyle='none', Marker='x', LineWidth=2, MarkerSize=10, DisplayName='Source Location')
        plot(ax2, start(1)-.5, start(2)-.5, LineStyle='none', Marker='.', LineWidth=2, MarkerSize=10, DisplayName='Start Location')
        hold(ax2, 'off')

        axis(ax2, 'equal')
        title(ax2, sprintf('Step %d', i))
        set(ax2, 'YDir', 'normal')
        set(ax2, 'XLim', [0 25])
        set(ax2, 'YLim', [0 25])

        legend(ax2, 'show', 'FontSize', 5, 'NumColumns', 1)

        plotInd = plotInd + 1;
    end
end

hold(ax1, 'off')
saveas(fig2, 'part1.png')

%% Part 2
map = ones(N, N)/(N^2);
S   = entropy(map);
traj = [];
maps = [];

x = start;

fig3 = figure(Position=[200 200 1000 1000]);
ax3 = axes(fig3);
set(ax3, 'YDir', 'normal')
set(ax3, 'XLim', [0 25])
set(ax3, 'YLim', [0 25])
hold(ax3, 'on')
axis(ax3, 'equal')

fig4 = figure(Position=[200, 200, 2000, 1000]);

set(fig3, 'Color', 'w')
set(fig4, 'Color', 'w')

i = 1;
plotInd = 1;
while S > 1e-2 && i < 1000
    traj(:, i) = x - [0.5; 0.5];
    measure = likelihood(s, x, true);
    z = rand() < measure;
    
    map = update_map(map, x, z);
    % x = move_random(x);
    S = entropy(map);
    x = move(map, x, s);
    maps(:, :, i) = map;

    cla(ax3)
    imagesc(ax3, xrange, yrange, map)
    colormap(ax3, 'hot')
    colorbar(ax3)
    mesh(ax3, xg, yg, zeros([N+1 N+1]), FaceColor='none', EdgeColor='k', DisplayName='Grid')
    plot(ax3, traj(1, :), traj(2, :), LineStyle='-', Marker='.', DisplayName='Robot Path')
    plot(ax3, s(1)-.5, s(2)-.5, LineStyle='none', Marker='x', LineWidth=2, MarkerSize=10, DisplayName='Source Location')
    plot(ax3, start(1)-.5, start(2)-.5, LineStyle='none', Marker='.', LineWidth=2, MarkerSize=10, DisplayName='Start Location')
    legend(ax3, 'show')
    drawnow

    i = i + 1;
    display(S)
end

plot_maps = round(linspace(1, i-1, 11));
for j = 1:10
    plotInd = plot_maps(j+1);

    ax4 = subplot(2, 5, j, 'Parent', fig4);

    hold(ax4, 'on')
    imagesc(ax4, xrange, yrange, maps(:, :, plotInd))
    colormap(ax4, 'hot')
    colorbar(ax4)
    mesh(ax4, xg, yg, zeros([N+1 N+1]), FaceColor='none', EdgeColor='k', DisplayName='Grid')
    plot(ax4, traj(1, 1:plotInd), traj(2, 1:plotInd), LineStyle='-', Marker='.', DisplayName='Robot Path')
    plot(ax4, s(1)-.5, s(2)-.5, LineStyle='none', Marker='x', LineWidth=2, MarkerSize=10, DisplayName='Source Location')
    plot(ax4, start(1)-.5, start(2)-.5, LineStyle='none', Marker='.', LineWidth=2, MarkerSize=10, DisplayName='Start Location')
    hold(ax4, 'off')

    title(ax4, sprintf('Step %d', plotInd))

    axis(ax4, 'equal')
    set(ax4, 'YDir', 'normal')
    set(ax4, 'XLim', [0 25])
    set(ax4, 'YLim', [0 25])

    legend(ax4, 'show', 'Location', 'best', 'FontSize', 5, 'NumColumns', 3)
end

saveas(fig4, 'part2_3.png')

%% Helper Functions
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

    % px = 0;
    % dx = 1/(row*col);

    % for i = 1:row
    %     for j = 1:col
    %         s = [j; i];
    %         pzx = likelihood(s, x, z);
    %         bx = map(i, j);
    %         px = px + pzx*bx*dx;
    %     end
    % end

    % b_total = 0;

    for i = 1:row
        for j = 1:col
            s = [j; i];
            bx = map(i, j);
            pzx = likelihood(s, x, z);
            bx_new = pzx*bx;
            map(i, j) = bx_new;
        end
    end

    map = map/evidence(map);

    map = map/sum(map, 'all');
end

function v = isValid(x)
    x1 = x(1);
    x2 = x(2);

    v = x1 > 0 && x1 < 26 && x2 > 0 && x2 < 26;
end

function x = move_random(x)
    i = 1;
    nextStep = x;
    
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

    x = nextStep(:, randInd);
end

function x_next = move(map, x, s)
    row = size(map, 1);
    col = size(map, 2);
    hx = entropy(map);
    x_next = x;

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

    dh_min = inf;
    for j = 1:i
        nextX = nextStep(:, j);
        
        map1 = update_map(map, nextX, true);
        map0 = update_map(map, nextX, false);

        h1 = entropy(map1);
        h0 = entropy(map0);

        l = likelihood(s, nextX, true);

        dx = 1/(row*col);
        p1 = 0;
        p0 = 0;
        for r = 1:row
            for c = 1:col
                s = [c; r];
                pzx0 = likelihood(s, nextX, false);
                pzx1 = likelihood(s, nextX, true);
                bx = map(i, j);
                p1 = p1 + pzx1*bx*dx;
                p0 = p0 + pzx0*bx*dx;
            end
        end

        dh1 = h1 - hx;
        dh0 = h0 - hx;

        dh = p0*dh0 + p1*dh1;

        if dh < dh_min
            x_next = nextX;
            dh_min = dh;
        end
        
        
    end

    display(dh_min)
    display(x_next)

end

function hx = entropy(map)
    row = size(map, 1);
    col = size(map, 2);

    hx = 0;
    
    for i = 1:row
        for j = 1:col
            bx = map(i, j);
            if bx > 1e-10
                hx = hx - bx*log(bx);
            end
        end
    end
end

function px = evidence(map)
    row = size(map, 1);
    col = size(map, 2);

    dx = 1/(row*col);
    px = sum(map, 'all')*dx;
end