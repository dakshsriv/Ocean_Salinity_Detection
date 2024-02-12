SELECT
    SUBSTR(c_date,0,11) AS c_date,
    t.t_lat,
    t.t_lon,
    AVG(c.c_value) AS v_chlorophyll
FROM chlorophyll c
INNER JOIN temperature_avg t ON 
    SUBSTR(c.c_date,0,11) = SUBSTR(t.t_date,0,11) AND
    ROUND(c.c_lat,1) = t.t_lat AND
    ROUND(c.c_lon,1) = t.t_lon
GROUP BY SUBSTR(c_date,0,11), t.t_lat, t.t_lon
ORDER BY SUBSTR(c_date,0,11), t.t_lat, t.t_lon
LIMIT 20;

INSERT INTO chlorophyll_avg 
SELECT
    SUBSTR(c_date,0,11) AS c_date,
    t.t_lat AS c_lat,
    t.t_lon AS c_lon,
    AVG(c.c_value) AS c_value
FROM chlorophyll c
INNER JOIN temperature_avg t ON 
    SUBSTR(c.c_date,0,11) = SUBSTR(t.t_date,0,11) AND
    ROUND(c.c_lat,1) = t.t_lat AND
    ROUND(c.c_lon,1) = t.t_lon
GROUP BY SUBSTR(c_date,0,11), t.t_lat, t.t_lon
ORDER BY SUBSTR(c_date,0,11), t.t_lat, t.t_lon
;

SELECT
    SUBSTR(s_date,0,11) AS v_date,
    s_lat+0.25 AS v_lat,
    s_lon+0.25 AS v_lon,
    t.t_value AS v_temperature,
    c.c_value AS v_chlorophyll,
    s_value AS v_salinity
FROM salinity s
INNER JOIN temperature_avg t ON 
    s.s_date = t.t_date AND
    s.s_lat+0.25 = t.t_lat AND
    s.s_lon+0.25 = t.t_lon
INNER JOIN chlorophyll_avg c ON 
    SUBSTR(c.c_date,0,11) = SUBSTR(s_date,0,11) AND
    ROUND(c.c_lat,1) = t.t_lat AND
    ROUND(c.c_lon,1) = t.t_lon
LIMIT 20;
