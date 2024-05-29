import React, { useState } from 'react';
import { Container, Alert, Spinner } from 'react-bootstrap';
import SearchForm from './components/SearchForm';
import JobList from './components/JobList';
import 'bootstrap/dist/css/bootstrap.min.css';

const App = () => {
  const [jobs, setJobs] = useState([]);
  const [error, setError] = useState();
  const [isLoading, setLoading] = useState(false);

  const handleSearch = async (searchParams) => {
    console.log(searchParams)
    setLoading(true)
    const fetchedJobs = await new Promise((resolve) => {
      setTimeout(() => {
        resolve([
          {
            title: 'Frontend Developer',
            company: 'Company A',
            location: 'New York, NY',
            description: 'Looking for a React developer...',
          },
          {
            title: 'Backend Developer',
            company: 'Company B',
            location: 'San Francisco, CA',
            description: 'Looking for a Node.js developer...',
          },
        ]);
      }, 2000);
    });
    try {
      
      setJobs(fetchedJobs);
    } catch(err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
    
  };

  return (
    <Container className="mt-5">
      <h1 className="mb-4">Job Recommendation System</h1>
      {error && <Alert variant="danger">{error}</Alert>}
      <SearchForm onSearch={handleSearch} />
      {isLoading ? (
        <div className="text-center my-4">
          <Spinner animation="border" />
        </div>
      ) : (
        <JobList jobs={jobs} />
      )}
    </Container>
  );
};

export default App;