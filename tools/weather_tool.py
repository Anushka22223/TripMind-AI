from tools.tavily_tool import tavily_search


def get_weather(destination):

    query = f"""
    Current weather forecast and travel weather conditions in {destination}
    """

    return tavily_search(query)