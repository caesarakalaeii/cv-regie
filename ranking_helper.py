from person_face_linker import Person, get_identity


class RankingHelper(object):

    def __init__(self, max_res: tuple[int, int]) -> None:
        self.max_res = max_res

    def get_amount_amount_of_faces(self, persons: list[Person]):
        faces = []
        if len(persons) == 0:
            return 0, faces
        for person in persons:
            identity = get_identity(person)
            if identity is "":
                continue
            if identity in faces:
                continue
            faces.append(identity)
        return len(faces), faces
