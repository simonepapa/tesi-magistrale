function Methodology() {
  return (
    <div className="mt-8 mb-12 flex flex-col gap-2 px-4 xl:mx-12 xl:max-w-[1200px] xl:px-0">
      <h1 className="mb-4 text-2xl font-bold">Methodology</h1>
      <p>
        The Crime Risk Index (CRI) is given by the summatory of all the crimes
        multiplied by their frequency and weight <br />
        (CRI = sum(crime_frequency * crime_weight)).
        <br /> It is then scaled into a [0, 100] range.
      </p>
      <p>
        The crimes were given these weights based on the damage (physical and
        psychological) inflicted on the victims as well as the potential
        long-term consequences of each crime:
      </p>
      <ul className="ml-6 list-disc list-none space-y-1 text-sm">
        <li>1.0 Murder</li>
        <li>0.7 Manslaughter</li>
        <li>0.8 Road Homicide</li>
        <li>0.9 Attempted Murder</li>
        <li>0.2 Theft</li>
        <li>0.7 Robbery</li>
        <li>0.8 Sexual violence</li>
        <li>0.6 Assault</li>
        <li>0.5 Drug Trafficking</li>
        <li>0.3 Fraud</li>
        <li>0.6 Extortion</li>
        <li>0.4 Smuggling</li>
        <li>1.0 Mafia-type association</li>
      </ul>
      <p>
        Since the number of articles highly varies between neighborhoods and the
        since the neighborhoods have different population, the CRI can
        optionally be weighted for these two factors. If so, then it will be
        divided by the number of crime per neighborhood and/or by the number of
        people in the neighborhood.
        <br />
        Note that, if the user decides to weigh by population, then the crime
        index will be multiplied by 10.000.
      </p>
    </div>
  );
}
export default Methodology;
