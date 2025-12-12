function useWindowsDimensions() {
  const { innerWidth: width, innerHeight: height } = window;

  return {
    width,
    height
  };
}

export { useWindowsDimensions };
