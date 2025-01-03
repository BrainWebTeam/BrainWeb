import React from 'react';
import CodeBlock from '../../CodeBlock';

const AmadeusTools: React.FC = () => {
  return (
    <section id="amadeus-tools">
      <h2>Amadeus Tools</h2>
      <p>
        Tools for integrating with the Amadeus API to access flight data, hotel bookings,
        and travel-related services.
      </p>

      <h3>Flight Search</h3>
      <CodeBlock
        language="typescript"
        code={`interface FlightSearchConfig {
  search: {
    origin: string;
    destination: string;
    departureDate: string;
    returnDate?: string;
    adults: number;
    children?: number;
    travelClass?: 'ECONOMY' | 'PREMIUM_ECONOMY' | 'BUSINESS' | 'FIRST';
  };
  filters: {
    maxPrice?: number;
    airlines?: string[];
    maxStops?: number;
    connectionTime?: {
      min: number;
      max: number;
    };
  };
  sort: {
    by: 'price' | 'duration' | 'stops';
    order: 'asc' | 'desc';
  };
}

class FlightSearch {
  async searchFlights(config: FlightSearchConfig) {
    // Initialize Amadeus client
    const amadeus = await this.initializeClient();
    
    // Search flights
    const results = await amadeus.shopping.flightOffers.get({
      originLocationCode: config.search.origin,
      destinationLocationCode: config.search.destination,
      departureDate: config.search.departureDate,
      returnDate: config.search.returnDate,
      adults: config.search.adults,
      children: config.search.children,
      travelClass: config.search.travelClass,
      ...this.buildFilters(config.filters)
    });
    
    // Sort results
    return this.sortResults(results, config.sort);
  }
}`}
      />

      <h3>Hotel Search</h3>
      <CodeBlock
        language="typescript"
        code={`interface HotelSearchConfig {
  search: {
    cityCode: string;
    checkInDate: string;
    checkOutDate: string;
    roomQuantity: number;
    adults: number;
  };
  amenities: string[];
  ratings: number[];
  priceRange: {
    min?: number;
    max?: number;
  };
  coordinates?: {
    latitude: number;
    longitude: number;
    radius: number;
  };
}

class HotelSearch {
  async searchHotels(config: HotelSearchConfig) {
    const amadeus = await this.initializeClient();
    
    return amadeus.shopping.hotelOffers.get({
      cityCode: config.search.cityCode,
      checkInDate: config.search.checkInDate,
      checkOutDate: config.search.checkOutDate,
      roomQuantity: config.search.roomQuantity,
      adults: config.search.adults,
      amenities: config.amenities,
      ratings: config.ratings,
      priceRange: config.priceRange,
      ...config.coordinates
    });
  }
}`}
      />
    </section>
  );
};

export default AmadeusTools;