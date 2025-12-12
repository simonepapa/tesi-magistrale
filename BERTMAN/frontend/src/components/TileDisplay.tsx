type Props = {
  style: string;
};

function TileDisplay({ style }: Props) {
  return (
    <div
      className="h-[88px] w-[88px] bg-center bg-no-repeat"
      style={{ backgroundImage: `url(img/${style}_tile.png)` }}></div>
  );
}
export default TileDisplay;
