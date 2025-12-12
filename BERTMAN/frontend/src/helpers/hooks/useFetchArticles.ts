import { Article } from "../../types/global";
import { useState, useEffect, useCallback } from "react";

const useFetchArticles = (setIsLoading: (loading: boolean) => void) => {
  const [articles, setArticles] = useState<Article[] | null>(null);

  const fetchArticles = useCallback(async () => {
    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:3000/api/get-articles");
      if (!response.ok)
        throw new Error(`HTTP error! Status: ${response.status}`);

      const jsonData = await response.json();
      setArticles(jsonData);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Request error", error);
    }

    setIsLoading(false);
  }, [setIsLoading]);

  useEffect(() => {
    fetchArticles();
  }, [fetchArticles]);

  return { articles };
};

export default useFetchArticles;
