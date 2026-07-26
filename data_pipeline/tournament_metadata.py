"""
Static reference table for ATP tournament venues: host country, coordinates,
and altitude. There is no free API for this, so this is a hand-built lookup
covering the ~60 tournaments that repeat every year and make up the large
majority of tour-level matches (Grand Slams, Masters 1000s, and the most
common 500s/250s). Anything not in this table (one-off Davis Cup ties,
smaller/rotating 250s, exhibitions) falls back to "unknown" - the features
that use this table (home country advantage, altitude, travel distance)
will just be missing/NaN for those matches rather than wrong.

Coordinates are city-level (not court-level) and altitude is approximate -
precise enough for "is this a home game" and "is this notably high
altitude", not survey-grade.
"""

# name -> (host country IOC code, latitude, longitude, altitude in meters)
TOURNAMENT_METADATA = {
    # Grand Slams
    "Australian Open": ("AUS", -37.8136, 144.9631, 31),
    "Roland Garros": ("FRA", 48.8566, 2.3522, 35),
    "US Open": ("USA", 40.7128, -74.0060, 10),
    "Wimbledon": ("GBR", 51.5074, -0.1278, 11),

    # Masters 1000 / Finals
    "Indian Wells Masters": ("USA", 33.7175, -116.3009, 74),
    "Miami Masters": ("USA", 25.7617, -80.1918, 2),
    "Monte Carlo Masters": ("MCO", 43.7384, 7.4246, 65),
    "Rome Masters": ("ITA", 41.9028, 12.4964, 21),
    "Madrid Masters": ("ESP", 40.4168, -3.7038, 667),
    "Canada Masters": ("CAN", 43.6532, -79.3832, 76),
    "Cincinnati Masters": ("USA", 39.1031, -84.5120, 229),
    "Shanghai Masters": ("CHN", 31.2304, 121.4737, 4),
    "Paris Masters": ("FRA", 48.8566, 2.3522, 35),
    "ATP Finals": ("ITA", 45.0703, 7.6869, 239),

    # 500s
    "Washington": ("USA", 38.9072, -77.0369, 8),
    "Barcelona": ("ESP", 41.3851, 2.1734, 12),
    "Hamburg": ("GER", 53.5511, 9.9937, 6),
    "Dubai": ("UAE", 25.2048, 55.2708, 5),
    "Acapulco": ("MEX", 16.8531, -99.8237, 3),
    "Rotterdam": ("NED", 51.9244, 4.4777, 0),
    "Rio Open": ("BRA", -22.9068, -43.1729, 2),
    "Vienna": ("AUT", 48.2082, 16.3738, 171),
    "Basel": ("SUI", 47.5596, 7.5886, 260),
    "Beijing": ("CHN", 39.9042, 116.4074, 44),
    "Tokyo": ("JPN", 35.6762, 139.6503, 40),
    "Queen's Club": ("GBR", 51.5074, -0.1278, 11),
    "Halle": ("GER", 51.4826, 8.2320, 112),

    # 250s
    "Auckland": ("NZL", -36.8485, 174.7633, 20),
    "Adelaide": ("AUS", -34.9285, 138.6007, 50),
    "Brisbane": ("AUS", -27.4698, 153.0251, 27),
    "Doha": ("QAT", 25.2854, 51.5310, 10),
    "Delray Beach": ("USA", 26.4615, -80.0728, 5),
    "Montpellier": ("FRA", 43.6108, 3.8767, 27),
    "Buenos Aires": ("ARG", -34.6037, -58.3816, 25),
    "Marseille": ("FRA", 43.2965, 5.3698, 12),
    "Kitzbuhel": ("AUT", 47.4467, 12.3922, 762),
    "Munich": ("GER", 48.1351, 11.5820, 520),
    "Stuttgart": ("GER", 48.7758, 9.1829, 245),
    "Eastbourne": ("GBR", 50.7687, 0.2713, 6),
    "Bastad": ("SWE", 56.4247, 12.8508, 15),
    "Umag": ("CRO", 45.4358, 13.5252, 5),
    "Gstaad": ("SUI", 46.4667, 7.2833, 1050),
    "Metz": ("FRA", 49.1193, 6.1757, 173),
    "Geneva": ("SUI", 46.2044, 6.1432, 375),
    "Winston-Salem": ("USA", 36.0999, -80.2442, 288),
    "Los Cabos": ("MEX", 22.8905, -109.9167, 4),
    "Cordoba": ("ARG", -31.4201, -64.1888, 360),
    "Santiago": ("CHI", -33.4489, -70.6693, 520),
    "Estoril": ("POR", 38.7071, -9.3979, 20),
    "Lyon": ("FRA", 45.7640, 4.8357, 173),
    "'s-Hertogenbosch": ("NED", 51.6878, 5.3037, 5),
    "Marrakech": ("MAR", 31.6295, -7.9811, 466),
    "Pune": ("IND", 18.5204, 73.8567, 560),
    "Sydney": ("AUS", -33.8688, 151.2093, 19),
    "Chengdu": ("CHN", 30.5728, 104.0668, 500),
    "Zhuhai": ("CHN", 22.2707, 113.5767, 8),
    "Antwerp": ("BEL", 51.2194, 4.4025, 6),
    "Moselle": ("FRA", 49.1193, 6.1757, 173),
    "Newport": ("USA", 41.4901, -71.3128, 6),
    "Atlanta": ("USA", 33.7490, -84.3880, 320),
    "Los Angeles": ("USA", 34.0522, -118.2437, 71),
    "Houston": ("USA", 29.7604, -95.3698, 13),
    "Dallas": ("USA", 32.7767, -96.7970, 131),
    "Indian Wells": ("USA", 33.7175, -116.3009, 74),
}

HIGH_ALTITUDE_THRESHOLD_M = 500


def get_tournament_metadata(tourney_name):
    """Return (country, lat, lon, altitude_m) or (None, None, None, None) if unknown."""
    return TOURNAMENT_METADATA.get(tourney_name, (None, None, None, None))
