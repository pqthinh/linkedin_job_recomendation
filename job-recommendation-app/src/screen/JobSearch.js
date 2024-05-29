import React, { useState } from 'react';
import { Container, Row, Col, Form, InputGroup, FormControl, Button, Card, Image } from 'react-bootstrap';
import { motion } from 'framer-motion';
import axios from "axios"
const JobSearch = () => {
  const [skills, setSkills] = useState('');
  const [desiredSalary, setDesiredSalary] = useState('');
  const [jobTitle, setJobTitle] = useState('');
  const [location, setLocation] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const response = await axios.post('/api/recommend-jobs', {
        skills,
        desiredSalary,
        jobTitle,
        location
      });

      const data = response.data;
      setSearchResults(data.jobs);
      setError('');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="app-container">
      <div className="hero-section">
        <Container>
          <Row>
            <Col md={6}>
              <h1>Tìm kiếm việc làm</h1>
              <p>Hệ thống đề xuất việc làm phù hợp với nhu cầu của bạn</p>
            </Col>
            <Col md={6}>
              <Image src="https://picsum.photos/200/300" alt="Job search illustration" />
            </Col>
          </Row>
        </Container>
      </div>

      <div className="search-form-container">
        <Container>
          <Row>
            <Col md={12}>
              <Form onSubmit={handleSubmit}>
                <InputGroup className="mb-3">
                  <FormControl
                    placeholder="Kỹ năng"
                    aria-label="Kỹ năng"
                    aria-describedby="basic-addon2"
                    value={skills}
                    onChange={(e) => setSkills(e.target.value)}
                  />
                </InputGroup>

                <InputGroup className="mb-3">
                  <FormControl
                    placeholder="Mức lương mong muốn"
                    aria-label="Mức lương mong muốn"
                    aria-describedby="basic-addon2"
                    value={desiredSalary}
                    onChange={(e) => setDesiredSalary(e.target.value)}
                  />
                </InputGroup>

                <InputGroup className="mb-3">
                  <FormControl
                    placeholder="Vị trí công việc"
                    aria-label="Vị trí công việc"
                    aria-describedby="basic-addon2"
                    value={jobTitle}
                    onChange={(e) => setJobTitle(e.target.value)}
                  />
                </InputGroup>

                <InputGroup className="mb-3">
                  <FormControl
                    placeholder="Nơi làm việc"
                    aria-label="Nơi làm việc"
                    aria-describedby="basic-addon2"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                  />
                </InputGroup>

                <Button variant="primary" type="submit">
                  Tìm kiếm
                </Button>
              </Form>
            </Col>
          </Row>
        </Container>
      </div>

      {error && <div className="error-message">{error}</div>}

      {searchResults.length > 0 && (
  <div className="results-container">
    <Container>
      <Row>
        {searchResults.map((job) => (
          <Col md={4} key={job.id}>
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, ease: 'easeInOut' }}
            >
              <Card className="job-card">
                <Card.Body>
                  <Card.Title>{job.title}</Card.Title>
                  <Card.Subtitle>{job.company}</Card.Subtitle>
                  <Card.Text>
                    <p>{job.location}</p>
                    <p>{job.salary}</p>
                  </Card.Text>
                  <Button variant="secondary" size="sm" href={job.url}>
                    Xem chi tiết
                  </Button>
                </Card.Body>
              </Card>
            </motion.div>
          </Col>
        ))}
      </Row>
    </Container>
  </div>
        )}
    </div>
  )

}
export default JobSearch